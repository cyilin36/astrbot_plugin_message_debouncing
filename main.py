import asyncio
from typing import Any, Dict, List, Optional

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Plain
from astrbot.api.star import Context, Star


class DebounceSession:
    """Represents an active debouncing session for a user or chat."""

    def __init__(self, key: str, initial_event: AstrMessageEvent, timeout: float) -> None:
        self.key = key
        self.initial_event = initial_event
        self.timeout = timeout
        self.collected_events: List[AstrMessageEvent] = []
        self.reset_event = asyncio.Event()
        self.cancelled = False
        self.lock = asyncio.Lock()


class MessageDebouncePlugin(Star):
    """AstrBot message debouncing plugin."""

    def __init__(
        self, context: Context, config: Optional[AstrBotConfig | Dict[str, Any]] = None
    ) -> None:
        super().__init__(context)
        self.config = config if config is not None else {}
        self.active_sessions: Dict[str, DebounceSession] = {}

    def _get_session_key(self, event: AstrMessageEvent) -> str:
        """Generate a unique key for the debouncing session.

        Group chats use platform:group:group_id:user_id so each user in the group
        has independent timing and buffer.
        Private chats use platform:private:user_id.
        """
        platform_id = event.get_platform_id() or event.get_platform_name() or "default"
        sender_id = str(event.get_sender_id())
        if event.is_private_chat():
            return f"{platform_id}:private:{sender_id}"
        group_id = str(event.get_group_id())
        return f"{platform_id}:group:{group_id}:{sender_id}"

    def _is_debouncing_enabled(self, event: AstrMessageEvent) -> bool:
        """Check if debouncing is enabled for this chat type."""
        if event.is_private_chat():
            return bool(self.config.get("enable_private", True))
        return bool(self.config.get("enable_group", True))

    def _get_timeout(self) -> float:
        """Get configured debouncing wait time in seconds."""
        val = self.config.get("debounce_seconds", 5.0)
        try:
            timeout = float(val)
            return max(0.1, timeout)
        except (ValueError, TypeError):
            return 5.0

    def _is_command_event(self, event: AstrMessageEvent) -> bool:
        """Check if the incoming message is a command.

        Commands should not be captured as follow-up text; they should flush or bypass debounce.
        """
        text = (event.message_str or "").strip()
        if text.startswith(("/", "／", "!", "！")):
            return True
        activated = event.get_extra("activated_handlers") or []
        for handler in activated:
            filters = getattr(handler, "event_filters", [])
            for f in filters:
                if f.__class__.__name__ == "CommandFilter":
                    return True
        return False

    @filter.event_message_type(filter.EventMessageType.ALL, priority=1000)
    async def intercept_follow_up(self, event: AstrMessageEvent) -> None:
        """High-priority event handler to intercept follow-up messages during debounce.

        Listening to EventMessageType.ALL ensures that follow-up messages in group chats
        (which may not include @bot) will not be dropped by WakingCheckStage.
        """
        if not self._is_debouncing_enabled(event):
            return

        session_key = self._get_session_key(event)
        session = self.active_sessions.get(session_key)
        if not session:
            return

        # If user sent a command during debounce, do not intercept as chat text.
        # Unblock the pending debounce session so the command can proceed.
        if self._is_command_event(event):
            logger.info(
                f"[Debounce] Command detected for {session_key}, flushing active debounce."
            )
            session.cancelled = True
            session.reset_event.set()
            return

        # Active debounce session exists for this user: capture message
        async with session.lock:
            session.collected_events.append(event)

            # Transfer temporary local files to initial event to preserve attachments
            temp_files = getattr(event, "_temporary_local_files", [])
            for path in list(temp_files):
                session.initial_event.track_temporary_local_file(path)
            temp_files.clear()

            # Signal reset event to restart countdown
            session.reset_event.set()

        logger.info(
            f"[Debounce] Captured follow-up message from {session_key}, resetting timer."
        )

        # Stop event propagation so this message does not trigger separate processing or LLM call
        event.stop_event()

    @filter.on_waiting_llm_request(priority=1000)
    async def debounce_waiting_llm(self, event: AstrMessageEvent) -> None:
        """Hook called when AstrBot determines this message will trigger LLM dialogue.

        Starts the debounce countdown, waits for follow-ups, and merges all collected
        messages once the timer expires.
        """
        if not self._is_debouncing_enabled(event):
            return

        session_key = self._get_session_key(event)
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
                    # No new messages within countdown window, debouncing complete
                    logger.info(
                        f"[Debounce] Timer expired for {session_key}, proceeding to LLM."
                    )
                    break
        finally:
            self.active_sessions.pop(session_key, None)

        # Merge collected messages into initial_event
        if not session.cancelled:
            self._merge_collected_events(session)

    def _merge_collected_events(self, session: DebounceSession) -> None:
        """Merge all collected messages into the initial event.

        Concatenates text and appends message components so AstrBot natively handles
        text, images, audio, and attachments without manual modality parsing.
        """
        if not session.collected_events:
            return

        initial_event = session.initial_event
        bot_self_id = str(initial_event.get_self_id())

        # 1. Merge text (message_str)
        text_lines: List[str] = []
        if initial_event.message_str and initial_event.message_str.strip():
            text_lines.append(initial_event.message_str.strip())

        for evt in session.collected_events:
            text = (evt.message_str or "").strip()
            if text:
                text_lines.append(text)

        initial_event.message_str = "\n".join(text_lines)

        # 2. Merge message components (message_obj.message)
        initial_chain = initial_event.get_messages()
        for evt in session.collected_events:
            evt_chain = evt.get_messages()
            if not evt_chain:
                continue
            # Add newline plain separator between distinct messages if initial chain has items
            if initial_chain:
                initial_chain.append(Plain("\n"))
            for comp in evt_chain:
                # Strip @bot from follow-up messages to keep prompt clean
                if isinstance(comp, At) and str(getattr(comp, "qq", "")) == bot_self_id:
                    continue
                initial_chain.append(comp)

        logger.info(
            f"[Debounce] Successfully merged {len(session.collected_events)} follow-up message(s) for {session.key}."
        )

    async def terminate(self) -> None:
        """Clean up any active debounce sessions when plugin is unloaded."""
        for session in list(self.active_sessions.values()):
            session.cancelled = True
            session.reset_event.set()
        self.active_sessions.clear()
