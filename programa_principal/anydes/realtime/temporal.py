from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, tzinfo
from typing import Callable, Literal
from zoneinfo import ZoneInfo


TemporalOperation = Literal["identity", "add", "subtract", "difference"]
TemporalUnit = Literal["minutes", "hours", "days", "weeks"]
TemporalOutput = Literal["time", "date", "datetime", "weekday", "day_period", "difference"]
TemporalReference = Literal["current_datetime", "explicit_range", "previous_temporal"]


@dataclass(frozen=True)
class TemporalQuery:
    reference: TemporalReference
    operation: TemporalOperation
    amount: int = 0
    unit: TemporalUnit = "minutes"
    timezone: str = ""
    output: TemporalOutput = "datetime"
    precision: str = "minute"
    clock_format: Literal["12h", "24h"] = "12h"
    start: datetime | None = None
    end: datetime | None = None
    relative_label: str = ""


@dataclass(frozen=True)
class TemporalResult:
    query: TemporalQuery
    reference_time: datetime
    value: datetime | None = None
    difference_seconds: float | None = None
    status: Literal["resolved", "needs_context", "invalid"] = "resolved"
    error: str = ""

    def as_data(self) -> dict[str, object]:
        return {
            "reference": self.query.reference,
            "operation": self.query.operation,
            "amount": self.query.amount,
            "unit": self.query.unit,
            "timezone": self.query.timezone,
            "output": self.query.output,
            "precision": self.query.precision,
            "clock_format": self.query.clock_format,
            "reference_time": self.reference_time.isoformat(),
            "value": self.value.isoformat() if self.value else "",
            "difference_seconds": self.difference_seconds,
            "status": self.status,
            "error": self.error,
        }


class TemporalQueryParser:
    """Parse general local-time operations without calculating their answer."""

    _weekdays = {
        "lunes": 0,
        "martes": 1,
        "miercoles": 2,
        "jueves": 3,
        "viernes": 4,
        "sabado": 5,
        "domingo": 6,
    }
    _number_words = {
        "un": 1,
        "una": 1,
        "uno": 1,
        "dos": 2,
        "tres": 3,
        "cuatro": 4,
        "cinco": 5,
        "seis": 6,
        "siete": 7,
        "ocho": 8,
        "nueve": 9,
        "diez": 10,
        "once": 11,
        "doce": 12,
        "trece": 13,
        "catorce": 14,
        "quince": 15,
        "dieciseis": 16,
        "diecisiete": 17,
        "dieciocho": 18,
        "diecinueve": 19,
        "veinte": 20,
        "veintiuno": 21,
        "veintidos": 22,
        "veintitres": 23,
        "veinticuatro": 24,
    }
    _units = {
        "minuto": "minutes",
        "minutos": "minutes",
        "hora": "hours",
        "horas": "hours",
        "dia": "days",
        "dias": "days",
        "semana": "weeks",
        "semanas": "weeks",
    }

    def parse(
        self,
        text: str,
        *,
        now: datetime,
        previous: TemporalResult | None = None,
    ) -> TemporalQuery | None:
        normalized = _normalize(text)
        if not self._is_temporal_request(text, normalized):
            return None
        timezone_name = _timezone_name(now.tzinfo)
        output = self._output(normalized)
        clock_format: Literal["12h", "24h"] = (
            "24h"
            if re.search(
                r"\b(?:formato|reloj)\s+(?:de\s+)?24\s*(?:horas?|h)?\b",
                normalized,
            )
            else "12h"
        )
        date_range = self._date_range(normalized, now)
        if date_range is not None:
            start, end = date_range
            unit = self._difference_unit(normalized)
            return TemporalQuery(
                reference="explicit_range",
                operation="difference",
                unit=unit,
                timezone=timezone_name,
                output="difference",
                precision=unit,
                clock_format=clock_format,
                start=start,
                end=end,
            )

        contextual = bool(
            re.search(
                r"\b(?:ese|esa|aquel|aquella)\s+(?:dia|noche|tarde|manana|momento)\b",
                normalized,
            )
            or re.search(r"\b(?:entonces|en ese momento)\b", normalized)
        )
        if contextual:
            return TemporalQuery(
                reference="previous_temporal",
                operation="identity",
                timezone=timezone_name,
                output=output,
                precision=self._precision(output),
                clock_format=clock_format,
                start=previous.value if previous and previous.value else None,
                relative_label="contextual",
            )

        named = self._named_relative(normalized)
        if named is not None:
            operation, amount, label = named
            return TemporalQuery(
                reference="current_datetime",
                operation=operation,
                amount=amount,
                unit="days",
                timezone=timezone_name,
                output=output if output != "time" or "hora" in normalized else "date",
                precision=self._precision(output),
                clock_format=clock_format,
                relative_label=label,
            )

        weekday = self._relative_weekday(normalized, now)
        if weekday is not None:
            return TemporalQuery(
                reference="current_datetime",
                operation="add",
                amount=weekday,
                unit="days",
                timezone=timezone_name,
                output="date" if output == "datetime" else output,
                precision="day",
                clock_format=clock_format,
                relative_label="next_weekday",
            )

        quantity = self._relative_quantity(normalized)
        if quantity is not None:
            operation, amount, unit = quantity
            return TemporalQuery(
                reference="current_datetime",
                operation=operation,
                amount=amount,
                unit=unit,
                timezone=timezone_name,
                output=output,
                precision=self._precision(output, unit),
                clock_format=clock_format,
                relative_label="relative_quantity",
            )

        return TemporalQuery(
            reference="current_datetime",
            operation="identity",
            timezone=timezone_name,
            output=output,
            precision=self._precision(output),
            clock_format=clock_format,
            relative_label="current",
        )

    @staticmethod
    def _is_temporal_request(original: str, normalized: str) -> bool:
        temporal_subject = bool(
            re.search(
                r"\b(?:hora|fecha)\b|"
                r"\b(?:que|cual)\s+dia\b|"
                r"\bdia\s+(?:fue|era|es|sera|cae)\b|"
                r"\bdia\s+de\s+la\s+semana\b|"
                r"\b(?:momento|parte|periodo)\s+del\s+dia\b|"
                r"\bcuant[oa]s?\s+(?:minutos?|horas?|dias?|semanas?)\b",
                normalized,
            )
        )
        if re.search(
            r"\b(?:mensaje|conversacion|pregunta|dije|dijiste|hablamos|"
            r"recuerdas|recordar)\b",
            normalized,
        ):
            return False
        requesting = bool(
            "?" in original
            or "¿" in original
            or re.search(
                r"\b(?:que|cual|cuando|cuant[oa]s?|dime|indica|"
                r"calcula|averigua|en que)\b",
                normalized,
            )
        )
        return temporal_subject and requesting

    @staticmethod
    def _output(normalized: str) -> TemporalOutput:
        if re.search(r"\bcuant[oa]s?\b.*\b(?:entre|desde)\b", normalized):
            return "difference"
        if re.search(r"\b(?:momento|parte|periodo)\s+del\s+dia\b", normalized):
            return "day_period"
        if re.search(r"\bdia\s+de\s+la\s+semana\b", normalized):
            return "weekday"
        has_hour = bool(re.search(r"\bhora\b", normalized))
        has_date = bool(re.search(r"\b(?:fecha|que\s+dia|cual\s+dia)\b", normalized))
        if has_hour and has_date:
            return "datetime"
        if has_hour:
            return "time"
        if has_date or re.search(
            r"\b(?:ayer|anteayer|manana|pasado\s+manana|"
            r"lunes|martes|miercoles|jueves|viernes|sabado|domingo)\b",
            normalized,
        ):
            return "date"
        return "datetime"

    def _relative_quantity(
        self,
        normalized: str,
    ) -> tuple[TemporalOperation, int, TemporalUnit] | None:
        number = r"(\d+|" + "|".join(self._number_words) + r")"
        unit = r"(minutos?|horas?|dias?|semanas?)"
        subtract = re.search(
            rf"\b(?:hace|atras|antes)\s+{number}\s+{unit}\b",
            normalized,
        )
        if subtract:
            return (
                "subtract",
                self._number(subtract.group(1)),
                self._units[subtract.group(2)],
            )
        add = re.search(
            rf"\b(?:dentro\s+de|despues\s+de|en)\s+{number}\s+{unit}\b",
            normalized,
        )
        if add:
            return (
                "add",
                self._number(add.group(1)),
                self._units[add.group(2)],
            )
        return None

    def _number(self, raw: str) -> int:
        return int(raw) if raw.isdigit() else self._number_words[raw]

    @staticmethod
    def _named_relative(
        normalized: str,
    ) -> tuple[TemporalOperation, int, str] | None:
        if re.search(r"\bpasado\s+manana\b", normalized):
            return "add", 2, "day_after_tomorrow"
        if re.search(r"\banteayer\b", normalized):
            return "subtract", 2, "day_before_yesterday"
        if re.search(r"\b(?:ayer|dia\s+anterior)\b", normalized):
            return "subtract", 1, "yesterday"
        tomorrow = bool(re.search(r"\bmanana\b", normalized)) and not bool(
            re.search(r"\b(?:por|de|en)\s+la\s+manana\b", normalized)
        )
        if tomorrow or re.search(r"\bdia\s+siguiente\b", normalized):
            return "add", 1, "tomorrow"
        if re.search(r"\bhoy\b", normalized):
            return "identity", 0, "today"
        return None

    def _relative_weekday(self, normalized: str, now: datetime) -> int | None:
        for name, weekday in self._weekdays.items():
            if not re.search(rf"\b{name}\b", normalized):
                continue
            if not re.search(
                rf"\b(?:proximo\s+{name}|{name}\s+que\s+viene|"
                rf"siguiente\s+{name})\b",
                normalized,
            ):
                continue
            delta = (weekday - now.weekday()) % 7
            return delta or 7
        return None

    def _date_range(
        self,
        normalized: str,
        now: datetime,
    ) -> tuple[datetime, datetime] | None:
        if not re.search(r"\b(?:entre|desde)\b", normalized):
            return None
        dates = [
            self._parse_date(match.group(0), now.tzinfo)
            for match in re.finditer(
                r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
                r"\d{1,2}[-/]\d{1,2}[-/]\d{4})"
                r"(?:[t ]\d{1,2}:\d{2})?\b",
                normalized,
            )
        ]
        valid = [item for item in dates if item is not None]
        if len(valid) >= 2:
            return valid[0], valid[1]
        times = [
            time(int(match.group(1)), int(match.group(2)))
            for match in re.finditer(r"\b(\d{1,2}):(\d{2})\b", normalized)
            if int(match.group(1)) < 24 and int(match.group(2)) < 60
        ]
        if len(times) < 2:
            return None
        start = datetime.combine(now.date(), times[0], tzinfo=now.tzinfo)
        end = datetime.combine(now.date(), times[1], tzinfo=now.tzinfo)
        if end < start:
            end += timedelta(days=1)
        return start, end

    @staticmethod
    def _parse_date(raw: str, zone: tzinfo | None) -> datetime | None:
        raw = raw.strip()
        try:
            if re.match(r"^\d{4}[-/]", raw):
                return datetime.fromisoformat(raw.replace("/", "-")).replace(
                    tzinfo=zone,
                )
            match = re.match(
                r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})"
                r"(?:[t ](\d{1,2}):(\d{2}))?$",
                raw,
            )
            if match is None:
                return None
            day, month, year = (int(item) for item in match.groups()[:3])
            hour = int(match.group(4) or 0)
            minute = int(match.group(5) or 0)
            return datetime(year, month, day, hour, minute, tzinfo=zone)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _difference_unit(normalized: str) -> TemporalUnit:
        if re.search(r"\bminutos?\b", normalized):
            return "minutes"
        if re.search(r"\bhoras?\b", normalized):
            return "hours"
        if re.search(r"\bsemanas?\b", normalized):
            return "weeks"
        return "days"

    @staticmethod
    def _precision(
        output: TemporalOutput,
        unit: TemporalUnit = "minutes",
    ) -> str:
        if output in {"date", "weekday"}:
            return "day"
        if output == "difference":
            return unit
        return "minute"


class RealtimeTemporalService:
    """Resolve and format verified local temporal facts without an LLM call."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        parser: TemporalQueryParser | None = None,
    ) -> None:
        self._clock = clock or local_now
        self._parser = parser or TemporalQueryParser()
        self._previous: TemporalResult | None = None

    def resolve(self, text: str) -> tuple[TemporalResult, str] | None:
        now = self._clock()
        if now.tzinfo is None:
            now = now.astimezone()
        query = self._parser.parse(text, now=now, previous=self._previous)
        if query is None:
            return None
        result = evaluate_temporal_query(
            query,
            now=now,
            previous=self._previous,
        )
        if result.status == "resolved":
            self._previous = result
        return result, format_temporal_result(result)


def evaluate_temporal_query(
    query: TemporalQuery,
    *,
    now: datetime,
    previous: TemporalResult | None = None,
) -> TemporalResult:
    if query.reference == "explicit_range":
        if query.start is None or query.end is None:
            return TemporalResult(
                query=query,
                reference_time=now,
                status="invalid",
                error="incomplete_explicit_range",
            )
        seconds = (query.end - query.start).total_seconds()
        return TemporalResult(
            query=query,
            reference_time=now,
            difference_seconds=seconds,
        )
    if query.reference == "previous_temporal":
        anchor = query.start or (previous.value if previous else None)
        if anchor is None:
            return TemporalResult(
                query=query,
                reference_time=now,
                status="needs_context",
                error="missing_temporal_anchor",
            )
        return TemporalResult(query=query, reference_time=now, value=anchor)

    delta = _delta(query.amount, query.unit)
    if query.operation == "subtract":
        value = now - delta
    elif query.operation == "add":
        value = now + delta
    else:
        value = now
    return TemporalResult(query=query, reference_time=now, value=value)


def format_temporal_result(result: TemporalResult) -> str:
    if result.status == "needs_context":
        return "¿A qué fecha o momento te refieres exactamente?"
    if result.status != "resolved":
        return "No pude interpretar con seguridad ese cálculo de tiempo."
    query = result.query
    if query.operation == "difference":
        seconds = abs(float(result.difference_seconds or 0.0))
        amount = _difference_amount(seconds, query.unit)
        return f"Hay {_number_text(amount, query.unit)} de diferencia."
    value = result.value
    if value is None:
        return "No pude obtener un momento válido."
    if query.output == "day_period":
        return _finish(f"En ese momento sería {_day_period(value.hour)}")
    if query.output == "weekday":
        return _finish(f"Sería {_date_text(value, include_weekday=True)}")
    if query.output == "date":
        phrase = _relative_date_prefix(query)
        return _finish(f"{phrase} {_date_text(value, include_weekday=True)}")
    if query.output == "datetime":
        return _finish(
            f"Sería {_date_text(value, include_weekday=True)}, "
            f"a las {_clock_text(value, query.clock_format)}"
        )
    if query.operation == "identity":
        return _finish(
            f"Ahora mismo son las {_clock_text(value, query.clock_format)}"
        )
    if query.operation == "subtract":
        return _finish(
            f"Hace {_number_text(query.amount, query.unit)} "
            f"eran las {_clock_text(value, query.clock_format)}"
        )
    return _finish(
        f"Dentro de {_number_text(query.amount, query.unit)} "
        f"serán las {_clock_text(value, query.clock_format)}"
    )


def local_now() -> datetime:
    configured = os.getenv("ANYDES_LOCAL_TIMEZONE", "").strip()
    if configured:
        try:
            return datetime.now(ZoneInfo(configured))
        except Exception:
            pass
    return datetime.now().astimezone()


def _delta(amount: int, unit: TemporalUnit) -> timedelta:
    if unit == "minutes":
        return timedelta(minutes=amount)
    if unit == "hours":
        return timedelta(hours=amount)
    if unit == "weeks":
        return timedelta(weeks=amount)
    return timedelta(days=amount)


def _difference_amount(seconds: float, unit: TemporalUnit) -> int | float:
    divisors = {
        "minutes": 60,
        "hours": 3600,
        "days": 86400,
        "weeks": 604800,
    }
    value = seconds / divisors[unit]
    return int(value) if value.is_integer() else round(value, 2)


def _number_text(amount: int | float, unit: TemporalUnit) -> str:
    labels = {
        "minutes": ("minuto", "minutos"),
        "hours": ("hora", "horas"),
        "days": ("día", "días"),
        "weeks": ("semana", "semanas"),
    }
    singular, plural = labels[unit]
    return f"{amount} {singular if amount == 1 else plural}"


def _clock_text(value: datetime, clock_format: str) -> str:
    if clock_format == "24h":
        return value.strftime("%H:%M")
    suffix = "a. m." if value.hour < 12 else "p. m."
    return f"{value.hour % 12 or 12}:{value.minute:02d} {suffix}"


def _date_text(value: datetime, *, include_weekday: bool) -> str:
    weekdays = (
        "lunes",
        "martes",
        "miércoles",
        "jueves",
        "viernes",
        "sábado",
        "domingo",
    )
    months = (
        "enero",
        "febrero",
        "marzo",
        "abril",
        "mayo",
        "junio",
        "julio",
        "agosto",
        "septiembre",
        "octubre",
        "noviembre",
        "diciembre",
    )
    prefix = f"{weekdays[value.weekday()]} " if include_weekday else ""
    return f"{prefix}{value.day} de {months[value.month - 1]} de {value.year}"


def _relative_date_prefix(query: TemporalQuery) -> str:
    labels = {
        "today": "Hoy es",
        "yesterday": "Ayer fue",
        "day_before_yesterday": "Anteayer fue",
        "tomorrow": "Mañana será",
        "day_after_tomorrow": "Pasado mañana será",
        "next_weekday": "Será",
    }
    if query.relative_label in labels:
        return labels[query.relative_label]
    if query.operation == "subtract":
        return f"Hace {_number_text(query.amount, query.unit)} fue"
    if query.operation == "add":
        return f"Dentro de {_number_text(query.amount, query.unit)} será"
    return "Es"


def _day_period(hour: int) -> str:
    if hour < 6:
        return "la madrugada"
    if hour < 12:
        return "la mañana"
    if hour < 19:
        return "la tarde"
    return "la noche"


def _timezone_name(zone: tzinfo | None) -> str:
    return getattr(zone, "key", None) or str(zone or "")


def _finish(text: str) -> str:
    return text if text.endswith((".", "?", "!")) else text + "."


def _normalize(text: str) -> str:
    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^a-z0-9./:-]+", " ", folded).split())
