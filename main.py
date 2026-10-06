import asyncio
import inspect
from typing import Any, Dict, List, Optional

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Plain
from astrbot.api.star import Context, Star


def _event_timestamp(event: AstrMessageEvent) -> float:
    """Best-effort arrival timestamp used to restore message order when merging."""
    try:
        return float(getattr(event, "created_at", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


class DebounceSession:
    """Represents an active debouncing session for one private chat user."""

    def __init__(self, key: str, initial_event: AstrMessageEvent, timeout: float) -> None:
        self.key = key
        self.initial_event = initial_event
        self.timeout = timeout
        self.collected_events: List[AstrMessageEvent] = []
        self.reset_event = asyncio.Event()
        self.cancelled = False
        self.closed = False
        """Set once the session stops accepting follow-ups (merged/popped/unloaded)."""
        self.processing = 0
        """Number of follow-ups currently being expanded; the countdown is held while > 0."""
        self.lock = asyncio.Lock()


class DebounceInterceptFilter(filter.CustomFilter):
    """Activate ``intercept_follow_up`` only while a debounce session is running.

    With this filter the interceptor stays dormant for every other message, so the
    plugin does not change how unrelated messages are processed.
    """

    plugin: Optional["MessageDebouncePlugin"] = None

    def __init__(self, raise_error: bool = True, **kwargs: Any) -> None:
        super().__init__(raise_error, **kwargs)

    def filter(self, event: AstrMessageEvent, cfg: AstrBotConfig) -> bool:
        plugin = DebounceInterceptFilter.plugin
        if plugin is None:
            return False
        try:
            return plugin.should_intercept(event)
        except Exception as e:  # never break the waking check with a filter error
            logger.warning(f"[Debounce] Intercept filter error: {e}", exc_info=True)
            return False


class MessageDebouncePlugin(Star):
    """AstrBot message debouncing plugin (private chats only)."""

    def __init__(
        self, context: Context, config: Optional[AstrBotConfig | Dict[str, Any]] = None
    ) -> None:
        super().__init__(context)
        self.config = config if config is not None else {}
        self.active_sessions: Dict[str, DebounceSession] = {}
        DebounceInterceptFilter.plugin = self

    # ------------------------------------------------------------------ config

    def _get_float(self, key: str, default: float, minimum: float = 0.0) -> float:
        val = self.config.get(key, default)
        try:
            out = float(val)
        except (TypeError, ValueError):
            return default
        if out != out:  # NaN
            return default
        return out if out >= minimum else minimum

    def _get_timeout(self) -> float:
        """Get configured debouncing wait time in seconds."""
        return max(0.1, self._get_float("debounce_seconds", 5.0, 0.1))

    def _get_preprocess_timeout(self) -> Optional[float]:
        """Timeout for follow-up preprocessing. None means no limit."""
        value = self._get_float("preprocess_timeout_seconds", 30.0, 0.0)
        return value if value > 0 else None

    @staticmethod
    def _handles_chat(event: AstrMessageEvent) -> bool:
        """Debouncing only ever applies to private chats; group chats are untouched.

        There is deliberately no plugin-level on/off switch: AstrBot's plugin page
        already provides enable/disable.
        """
        return event.is_private_chat()

    # ------------------------------------------------------------- session key

    def _get_session_key(self, event: AstrMessageEvent) -> str:
        """Generate a unique key for the debouncing session.

        ``unified_msg_origin`` identifies the private conversation; the sender is
        appended to stay safe on platforms where one origin can carry several
        senders.
        """
        sender_id = str(event.get_sender_id())
        scope = str(getattr(event, "unified_msg_origin", "") or "")
        if not scope or scope.endswith(":"):
            platform_id = event.get_platform_id() or event.get_platform_name() or "default"
            scope = f"{platform_id}:private"
        return f"{scope}::{sender_id}"

    # --------------------------------------------------------------- utilities

    def _is_live(self, session: DebounceSession) -> bool:
        """True when the session is still the registered, accepting session."""
        return not session.closed and self.active_sessions.get(session.key) is session

    def _get_live_session(self, key: str) -> Optional[DebounceSession]:
        session = self.active_sessions.get(key)
        if session is None or session.closed:
            return None
        return session

    @staticmethod
    def _transfer_temp_files(source: AstrMessageEvent, target: AstrMessageEvent) -> None:
        """Move event-scoped temporary files so the source cleanup cannot delete them."""
        files = getattr(source, "_temporary_local_files", None)
        if not files:
            return
        for path in list(files):
            try:
                target.track_temporary_local_file(path)
            except Exception as e:
                logger.warning(f"[Debounce] Failed to retain temporary file {path}: {e}")
        files.clear()

    # ------------------------------------------------------------- interception

    def should_intercept(self, event: AstrMessageEvent) -> bool:
        """Whether ``intercept_follow_up`` has something to do for this event."""
        if not self._handles_chat(event):
            return False
        return self._get_live_session(self._get_session_key(event)) is not None

    def _is_command_event(self, event: AstrMessageEvent) -> bool:
        """Check if the incoming message is a command.

        Commands must not be captured as follow-up text.  AstrBot only records
        command handlers whose ``CommandFilter`` matched, so that list is the
        authoritative signal (the wake prefix has already been stripped from
        ``message_str`` by then).
        """
        activated = event.get_extra("activated_handlers") or []
        for handler in activated:
            for f in getattr(handler, "event_filters", []) or []:
                if f.__class__.__name__ in ("CommandFilter", "CommandGroupFilter"):
                    return True
        text = (event.message_str or "").strip()
        return text.startswith(("/", "／"))

    def _is_preprocessor_handler(self, handler: Any) -> bool:
        """Check if an activated handler is a message preprocessor or content expander.

        For example: forward message readers (astrbot_plugin_forward_reader_person),
        voice-to-text converters, or nested attachment unpackers.
        """
        handler_name = getattr(handler, "handler_name", "")
        if handler_name in ("intercept_follow_up", "debounce_waiting_llm"):
            return False

        full_name = str(getattr(handler, "handler_full_name", "")).lower()
        name = str(handler_name).lower()
        module = str(getattr(handler, "handler_module_path", "")).lower()

        # Exclude debounce plugin itself
        if "debounce" in full_name or "debounce" in module:
            return False

        # Exclude commands
        filters = getattr(handler, "event_filters", []) or []
        for f in filters:
            if f.__class__.__name__ in ("CommandFilter", "CommandGroupFilter"):
                return False

        # Match known preprocessor keywords
        keywords = ("forward", "reader", "expand", "unpack", "preprocess", "normaliz")
        if any(k in full_name or k in name or k in module for k in keywords):
            return True

        # Match filters indicating forward or preprocessor functionality
        for f in filters:
            f_name = f.__class__.__name__.lower()
            if "forward" in f_name or "preprocess" in f_name:
                return True

        return False

    async def _preprocess_with_timeout(self, event: AstrMessageEvent) -> None:
        """Run follow-up preprocessors, bounded so a stuck one cannot stall the wait."""
        timeout = self._get_preprocess_timeout()
        if timeout is None:
            try:
                await self._preprocess_follow_up_event(event)
            except Exception as e:
                logger.error(
                    f"[Debounce] Follow-up preprocessing failed: {e}", exc_info=True
                )
            return
        try:
            await asyncio.wait_for(
                self._preprocess_follow_up_event(event), timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[Debounce] Follow-up preprocessing timed out after {timeout}s; "
                "merging the message without full expansion."
            )
        except Exception as e:
            # Never let a broken preprocessor cost the user their message.
            logger.error(
                f"[Debounce] Follow-up preprocessing failed: {e}", exc_info=True
            )

    async def _preprocess_follow_up_event(self, event: AstrMessageEvent) -> None:
        """Run preprocessor handlers (such as forward message expanders) on follow-up events.

        This ensures that forward messages, replies, and attachments in follow-up messages
        are fully expanded into Plain, Image, etc., before we intercept and merge them.

        Only handlers positioned *after* ``intercept_follow_up`` are run: handlers with a
        higher priority already ran in this event's pipeline, and calling them again would
        duplicate their work (e.g. expanding the same forward twice).
        """
        activated = list(event.get_extra("activated_handlers") or [])
        handlers_parsed_params = event.get_extra("handlers_parsed_params") or {}

        own_index: Optional[int] = None
        for index, handler in enumerate(activated):
            if getattr(handler, "handler_name", "") == "intercept_follow_up":
                own_index = index
                break
        candidates = activated[own_index + 1 :] if own_index is not None else activated

        executed = set()

        for handler in candidates:
            if not self._is_preprocessor_handler(handler):
                continue
            handler_func = getattr(handler, "handler", None)
            if not callable(handler_func):
                continue

            full_name = getattr(handler, "handler_full_name", str(handler))
            executed.add(full_name)
            params = handlers_parsed_params.get(full_name, {})
            try:
                logger.info(
                    f"[Debounce] Preprocessing follow-up message with {full_name}."
                )
                res = handler_func(event, **params)
                if inspect.isawaitable(res):
                    await res
                elif inspect.isasyncgen(res):
                    async for _ in res:
                        pass
            except Exception as e:
                logger.error(
                    f"[Debounce] Error running preprocessor {full_name}: {e}",
                    exc_info=True,
                )

        if not executed:
            await self._fallback_preprocess_forward_msg(event)

    @staticmethod
    def _handler_filters_pass(handler: Any, event: AstrMessageEvent, cfg: Any) -> bool:
        filters = getattr(handler, "event_filters", []) or []
        if not filters:
            return False
        for f in filters:
            try:
                if not f.filter(event, cfg):
                    return False
            except Exception:
                return False
        return True

    async def _fallback_preprocess_forward_msg(self, event: AstrMessageEvent) -> None:
        """Fallback to trigger forward reader handler from global registry if not in activated_handlers."""
        try:
            from astrbot.core.star.star_handler import star_handlers_registry, EventType
        except ImportError:
            return

        if not star_handlers_registry:
            return

        # Check if message contains forward or reply components
        message_obj = getattr(event, "message_obj", None)
        raw_msg = getattr(message_obj, "raw_message", None)
        comps = list(getattr(message_obj, "message", []) or [])

        has_forward_candidate = False
        for comp in comps:
            c_name = comp.__class__.__name__.lower()
            if c_name in ("forward", "node", "nodes", "reply"):
                has_forward_candidate = True
                break
        if not has_forward_candidate and isinstance(raw_msg, dict):
            raw_text = str(raw_msg).lower()
            if "forward" in raw_text or "node" in raw_text:
                has_forward_candidate = True

        if not has_forward_candidate:
            return

        cfg = None
        try:
            if self.context is not None:
                cfg = self.context.get_config(umo=event.unified_msg_origin)
        except Exception as e:
            logger.debug(f"[Debounce] Unable to load config for fallback filters: {e}")

        for handler in star_handlers_registry.get_handlers_by_event_type(
            EventType.AdapterMessageEvent
        ):
            if not self._is_preprocessor_handler(handler):
                continue
            handler_func = getattr(handler, "handler", None)
            if not callable(handler_func):
                continue
            if cfg is not None and not self._handler_filters_pass(handler, event, cfg):
                continue
            full_name = getattr(handler, "handler_full_name", str(handler))
            try:
                logger.info(
                    f"[Debounce] Fallback running preprocessor {full_name} on follow-up event."
                )
                res = handler_func(event)
                if inspect.isawaitable(res):
                    await res
                elif inspect.isasyncgen(res):
                    async for _ in res:
                        pass
            except Exception as e:
                logger.error(
                    f"[Debounce] Error in fallback preprocessor {full_name}: {e}",
                    exc_info=True,
                )

    # ---------------------------------------------------------------- handlers

    # NOTE: decorator order matters. `event_message_type` must be applied first so
    # that `get_handler_or_create` creates the handler metadata with priority=1000
    # (a later decorator would reuse that metadata and ignore its own kwargs).
    @filter.custom_filter(DebounceInterceptFilter)
    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE, priority=1000)
    async def intercept_follow_up(self, event: AstrMessageEvent) -> None:
        """High-priority handler that captures follow-up messages during debounce."""
        if not self._handles_chat(event):
            return

        session_key = self._get_session_key(event)
        session = self._get_live_session(session_key)
        if session is None:
            return

        # A command is not chat text: leave it to AstrBot's command handling and do
        # not cancel the session either, or the messages already buffered would be
        # lost.
        if self._is_command_event(event):
            logger.info(
                f"[Debounce] Command detected for {session_key}, leaving the "
                "debounce buffer untouched."
            )
            return

        if await self._absorb_event(session, event):
            logger.info(
                f"[Debounce] Captured follow-up message from {session_key}, "
                "resetting timer."
            )
            # Stop event propagation so this message does not trigger a separate
            # processing pass or LLM call.
            event.stop_event()

    async def _absorb_event(
        self,
        session: DebounceSession,
        event: AstrMessageEvent,
        *,
        preprocess: bool = True,
    ) -> bool:
        """Append ``event`` to ``session``'s buffer. Returns False if it did not fit.

        When it returns False the event was left untouched (flags restored) so the
        caller can let it continue through the normal pipeline instead of silently
        swallowing it.
        """
        previous_flags = (event.is_wake, event.is_at_or_wake_command)
        # Follow-up events must look woken for preprocessors that respect the
        # bot's wake rules (e.g. forward_reader's trigger_mode="respect_astrbot").
        event.is_wake = True
        event.is_at_or_wake_command = True

        async with session.lock:
            if not self._is_live(session):
                event.is_wake, event.is_at_or_wake_command = previous_flags
                return False
            # Reset the countdown before any slow preprocessing and hold it while
            # the follow-up is being expanded, so the session cannot expire in the
            # middle of the work and drop the message afterwards.
            session.processing += 1
            session.reset_event.set()

        absorbed = False
        try:
            if preprocess:
                await self._preprocess_with_timeout(event)
            async with session.lock:
                if self._is_live(session):
                    session.collected_events.append(event)
                    self._transfer_temp_files(event, session.initial_event)
                    session.reset_event.set()
                    absorbed = True
        finally:
            async with session.lock:
                session.processing -= 1

        if not absorbed:
            event.is_wake, event.is_at_or_wake_command = previous_flags
            return False

        event.set_extra("debounce_captured", True)
        return True

    @filter.on_waiting_llm_request(priority=1000)
    async def debounce_waiting_llm(self, event: AstrMessageEvent) -> None:
        """Hook called when AstrBot determines this message will trigger LLM dialogue.

        Starts the debounce countdown, waits for follow-ups, and merges all collected
        messages once the timer expires.
        """
        if not self._handles_chat(event):
            return

        session_key = self._get_session_key(event)

        # A concurrent message from the same user may already be debouncing (messages
        # are dispatched as independent tasks). Absorb it instead of overwriting the
        # running session, which would produce two LLM replies.
        existing = self._get_live_session(session_key)
        if existing is not None and await self._absorb_event(
            existing, event, preprocess=False
        ):
            logger.info(
                f"[Debounce] Merged a concurrent message into the active session "
                f"{session_key}."
            )
            event.stop_event()
            return

        timeout = self._get_timeout()
        session = DebounceSession(session_key, event, timeout)
        self.active_sessions[session_key] = session
        logger.info(
            f"[Debounce] Started debouncing session for {session_key} with timeout {timeout}s."
        )

        try:
            current_timeout = timeout
            while True:
                session.reset_event.clear()
                try:
                    await asyncio.wait_for(
                        session.reset_event.wait(), timeout=current_timeout
                    )
                    if session.cancelled:
                        break
                    # Reset timeout countdown
                    current_timeout = self._get_timeout()
                    logger.debug(
                        f"[Debounce] Timer reset to {current_timeout}s for {session_key}."
                    )
                except asyncio.TimeoutError:
                    # A follow-up is still being expanded: hold the session instead of
                    # merging without it, otherwise a slow preprocessor would silently
                    # drop the message it was working on.
                    async with session.lock:
                        if session.processing > 0 and not session.cancelled:
                            current_timeout = self._get_timeout()
                            logger.debug(
                                f"[Debounce] Holding the session for {session_key} "
                                "while a follow-up is being preprocessed."
                            )
                            continue
                    # No new messages within countdown window, debouncing complete
                    logger.info(
                        f"[Debounce] Timer expired for {session_key}, proceeding to LLM."
                    )
                    break
        finally:
            async with session.lock:
                session.closed = True
                if self.active_sessions.get(session_key) is session:
                    self.active_sessions.pop(session_key, None)

        # Merge whatever was collected. Buffered follow-ups were already stopped, so
        # dropping them here would lose the user's text for good.
        self._merge_collected_events(session)

    def _merge_collected_events(self, session: DebounceSession) -> None:
        """Merge all collected messages into the initial event.

        Concatenates text and appends message components so AstrBot natively handles
        text, images, audio, and attachments without manual modality parsing.
        Messages are ordered by arrival time, which keeps the prompt in the order the
        user actually sent it even when a later message opened the session first.
        """
        initial_event = session.initial_event
        events = [initial_event] + list(session.collected_events)
        if len(events) <= 1:
            return

        indexed = list(enumerate(events))
        indexed.sort(key=lambda item: (_event_timestamp(item[1]), item[0]))
        ordered = [evt for _, evt in indexed]

        bot_self_id = str(initial_event.get_self_id())

        # 1. Merge text (message_str)
        text_lines: List[str] = []
        for evt in ordered:
            text = (evt.message_str or "").strip()
            if text:
                text_lines.append(text)
        initial_event.message_str = "\n".join(text_lines)

        # 2. Merge message components (message_obj.message)
        merged_chain: List[Any] = []
        for index, evt in enumerate(ordered):
            evt_chain = evt.get_messages()
            if not evt_chain:
                continue
            if index == 0:
                valid_comps = list(evt_chain)
            else:
                # Strip @bot from follow-up messages to keep the prompt clean
                valid_comps = [
                    comp
                    for comp in evt_chain
                    if not (
                        isinstance(comp, At)
                        and str(getattr(comp, "qq", "")) == bot_self_id
                    )
                ]
            if not valid_comps:
                continue
            # Add newline plain separator between distinct messages
            if merged_chain:
                merged_chain.append(Plain("\n"))
            merged_chain.extend(valid_comps)

        message_obj = getattr(initial_event, "message_obj", None)
        if message_obj is not None:
            message_obj.message_str = initial_event.message_str
            if merged_chain:
                message_obj.message = merged_chain

        logger.info(
            f"[Debounce] Successfully merged {len(session.collected_events)} "
            f"follow-up message(s) for {session.key}."
        )

    async def terminate(self) -> None:
        """Clean up any active debounce sessions when plugin is unloaded."""
        for session in list(self.active_sessions.values()):
            session.cancelled = True
            session.reset_event.set()
        self.active_sessions.clear()
