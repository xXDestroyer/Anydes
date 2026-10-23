from __future__ import annotations

import json
import os
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from anydes.memoria.data_protection import DataProtector, default_data_protector
from anydes.nucleo.path_locks import shared_path_lock


TaskStatus = Literal[
    "proposed",
    "awaiting_confirmation",
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
]


@dataclass(frozen=True)
class RealtimeTask:
    task_id: str
    objective: str
    origin_event_id: str
    tool: str
    parameters: dict[str, object]
    permissions: tuple[str, ...]
    status: TaskStatus
    created_at: str
    started_at: str = ""
    completed_at: str = ""
    result: dict[str, object] = field(default_factory=dict)
    error: str = ""
    delivery_destination: str = "conversation"


@dataclass(frozen=True)
class TaskNotification:
    task_id: str
    status: Literal["completed", "failed", "cancelled"]
    objective: str
    result: dict[str, object] = field(default_factory=dict)
    error_type: str = ""
    delivery_destination: str = "conversation"

    def visible_text(self) -> str:
        if self.status == "completed":
            message = str(self.result.get("message") or "").strip()
            return message or f"Ya terminé: {self.objective}."
        message = str(self.result.get("message") or "").strip()
        if message:
            return message
        if self.status == "cancelled":
            return f"Cancelé la tarea: {self.objective}."
        return f"No pude completar la tarea: {self.objective}."


class TaskManager:
    """Owns asynchronous work and publishes verified completion notifications."""

    def __init__(
        self,
        path: Path,
        *,
        protector: DataProtector | None = None,
        max_workers: int = 2,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._protector = protector or default_data_protector(
            passphrase=os.getenv("ANYDES_MEMORY_PASSPHRASE")
        )
        self._lock = threading.RLock()
        self._path_lock = shared_path_lock(self.path)
        self._tasks, self.load_error = self._load()
        self._futures: dict[str, Future] = {}
        self._notifications: list[TaskNotification] = []
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers),
            thread_name_prefix="anydes-realtime-task",
        )
        self._closed = False

    def propose(
        self,
        *,
        objective: str,
        origin_event_id: str,
        tool: str,
        parameters: dict[str, object] | None = None,
        permissions: tuple[str, ...] = (),
        requires_confirmation: bool = False,
        delivery_destination: str = "conversation",
    ) -> RealtimeTask:
        self._ensure_writable()
        if not objective.strip() or not origin_event_id.strip() or not tool.strip():
            raise ValueError("A task requires objective, origin event, and tool.")
        now = _utc_now()
        task = RealtimeTask(
            task_id=f"realtime-task:{uuid.uuid4().hex}",
            objective=objective.strip(),
            origin_event_id=origin_event_id,
            tool=tool.strip(),
            parameters=dict(parameters or {}),
            permissions=tuple(permissions),
            status="awaiting_confirmation" if requires_confirmation else "proposed",
            created_at=now,
            delivery_destination=delivery_destination,
        )
        with self._lock:
            self._tasks[task.task_id] = task
            self._persist_locked()
        return task

    def queue(
        self,
        task_id: str,
        operation: Callable[[], dict[str, object]],
    ) -> RealtimeTask:
        self._ensure_writable()
        with self._lock:
            task = self._require(task_id)
            if task.status not in {"proposed", "awaiting_confirmation"}:
                raise ValueError(f"Task {task_id} cannot be queued from {task.status}.")
            queued = replace(task, status="queued")
            self._tasks[task_id] = queued
            self._persist_locked()
            future = self._executor.submit(self._run, task_id, operation)
            self._futures[task_id] = future
            return queued

    def confirm(
        self,
        task_id: str,
        operation: Callable[[], dict[str, object]],
    ) -> RealtimeTask:
        with self._lock:
            task = self._require(task_id)
            if task.status != "awaiting_confirmation":
                raise ValueError(
                    f"Task {task_id} is not awaiting confirmation."
                )
        return self.queue(task_id, operation)

    def cancel(self, task_id: str) -> RealtimeTask:
        self._ensure_writable()
        with self._lock:
            task = self._require(task_id)
            if task.status in {"completed", "failed", "cancelled"}:
                return task
            future = self._futures.get(task_id)
            if future is not None:
                future.cancel()
            cancelled = replace(
                task,
                status="cancelled",
                completed_at=_utc_now(),
            )
            self._tasks[task_id] = cancelled
            self._notifications.append(
                TaskNotification(
                    task_id=task_id,
                    status="cancelled",
                    objective=task.objective,
                    delivery_destination=task.delivery_destination,
                )
            )
            self._persist_locked()
            return cancelled

    def get(self, task_id: str) -> RealtimeTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list(self, *, status: TaskStatus | None = None) -> tuple[RealtimeTask, ...]:
        with self._lock:
            tasks = tuple(self._tasks.values())
        if status is None:
            return tasks
        return tuple(task for task in tasks if task.status == status)

    def drain_notifications(self) -> tuple[TaskNotification, ...]:
        with self._lock:
            items = tuple(self._notifications)
            self._notifications.clear()
            return items

    def close(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _run(
        self,
        task_id: str,
        operation: Callable[[], dict[str, object]],
    ) -> None:
        with self._lock:
            task = self._require(task_id)
            if task.status == "cancelled":
                return
            running = replace(task, status="running", started_at=_utc_now())
            self._tasks[task_id] = running
            self._persist_locked()
        try:
            result = dict(operation() or {})
        except Exception as exc:
            with self._lock:
                current = self._require(task_id)
                failed = replace(
                    current,
                    status="failed",
                    completed_at=_utc_now(),
                    error=type(exc).__name__,
                )
                self._tasks[task_id] = failed
                self._notifications.append(
                    TaskNotification(
                        task_id=task_id,
                        status="failed",
                        objective=current.objective,
                        error_type=type(exc).__name__,
                        delivery_destination=current.delivery_destination,
                    )
                )
                self._persist_locked()
            return
        if result.get("_task_success") is False:
            with self._lock:
                current = self._require(task_id)
                failed = replace(
                    current,
                    status="failed",
                    completed_at=_utc_now(),
                    result=result,
                    error=str(result.get("_task_error") or "operation_not_verified"),
                )
                self._tasks[task_id] = failed
                self._notifications.append(
                    TaskNotification(
                        task_id=task_id,
                        status="failed",
                        objective=current.objective,
                        result=result,
                        error_type=failed.error,
                        delivery_destination=current.delivery_destination,
                    )
                )
                self._persist_locked()
            return
        with self._lock:
            current = self._require(task_id)
            if current.status == "cancelled":
                return
            completed = replace(
                current,
                status="completed",
                completed_at=_utc_now(),
                result=result,
                error="",
            )
            self._tasks[task_id] = completed
            self._notifications.append(
                TaskNotification(
                    task_id=task_id,
                    status="completed",
                    objective=current.objective,
                    result=result,
                    delivery_destination=current.delivery_destination,
                )
            )
            self._persist_locked()

    def _require(self, task_id: str) -> RealtimeTask:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Unknown task: {task_id}")
        return task

    def _persist_locked(self) -> None:
        self._ensure_writable()
        payload = json.dumps(
            [asdict(task) for task in self._tasks.values()],
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        envelope = {
            "version": 1,
            "scheme": self._protector.scheme,
            "payload": self._protector.protect(payload),
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with self._path_lock:
            temporary.write_text(
                json.dumps(envelope, ensure_ascii=True, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)

    def _load(self) -> tuple[dict[str, RealtimeTask], bool]:
        if not self.path.exists():
            return {}, False
        try:
            with self._path_lock:
                envelope = json.loads(self.path.read_text(encoding="utf-8"))
            raw = self._protector.unprotect(str(envelope["payload"]))
            items = json.loads(raw.decode("utf-8"))
        except Exception:
            return {}, True
        tasks: dict[str, RealtimeTask] = {}
        for payload in items:
            payload["parameters"] = dict(payload.get("parameters") or {})
            payload["permissions"] = tuple(payload.get("permissions") or ())
            payload["result"] = dict(payload.get("result") or {})
            task = RealtimeTask(**payload)
            if task.status in {"queued", "running"}:
                task = replace(
                    task,
                    status="failed",
                    completed_at=_utc_now(),
                    error="interrupted_by_restart",
                )
            tasks[task.task_id] = task
        return tasks, False

    def _ensure_writable(self) -> None:
        if self.load_error:
            raise RuntimeError(
                "Existing task state could not be read; it was preserved."
            )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
