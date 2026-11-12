"""에이전트가 호출하는 도메인 도구와, 도구가 쓰는 검증·저장 로직.

- 1단계: search_regulations (규정 검색)
- 3단계: find_host (피방문자 확인), create_visit_request (검증 → 확정 대기 → 저장)
- 5단계: get_my_requests (이름 + 휴대폰번호로 본인 신청 목록), get_request_detail (목록 번호로 상세),
          cancel_my_request (전체 신청번호로 즉시 취소)

[Java 개발자를 위한 개요]
- "도구(tool)"는 LLM 이 필요할 때 호출해 달라고 요청하는 일반 Python 함수다.
- @tool 데코레이터(Java 의 애너테이션과 비슷)를 붙이면, LangChain 이 함수 이름·인자 타입·독스트링을 읽어
  LLM 에게 "이런 도구가 있고 이렇게 부르면 된다" 는 설명(JSON 스키마)을 자동으로 만들어 준다.
  그래서 도구의 독스트링이 곧 LLM 이 읽는 도구 설명서다. 내용을 바꾸면 LLM 이 도구를 고르는 방식도 달라진다.
- 신청 필드 값은 LLM 이 인자로 넘기지 않는다. LCEL 구조화 추출(agent.py)이 그래프 state 에 넣어 둔 값을
  도구가 직접 읽는다(CLAUDE.md: 수집 필드의 기준값은 그래프 state). LLM 이 값을 잘못 옮겨 적는 일을 막기 위해서다.
"""

import json
import os
import re
import threading  # 스레드 잠금 (java.util.concurrent.locks.ReentrantLock 과 비슷한 threading.Lock)
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from langchain.tools import ToolRuntime  # 도구 안에서 그래프 state 와 tool_call_id 를 읽게 해 주는 객체
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command, interrupt  # Command: state 갱신 명령, interrupt: 그래프를 멈추고 사람 입력 기다리기

from src import retriever

# ── 데이터 경로 ───────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMPLOYEES_PATH = DATA_DIR / "employees.json"  # 더미 임직원 명단 (이름, 부서)
SEED_REQUESTS_PATH = DATA_DIR / "seed_requests.json"  # 커밋된 시드 신청 (읽기 전용, 본인 조회 테스트용)
# 실행 중 생기는 신청 저장소. .gitignore 대상이다. 테스트는 이 변수를 임시 경로로 바꿔 작업 폴더를 건드리지 않는다.
REQUESTS_PATH = DATA_DIR / "runtime" / "requests.json"

# 한국 표준시. KST 는 서머타임이 없어서 고정 +9시간으로 충분하다
# (Windows 에서는 ZoneInfo("Asia/Seoul") 을 쓰려면 tzdata 패키지가 따로 필요해서 고정 오프셋을 쓴다).
KST = timezone(timedelta(hours=9))

# 신청 필드 이름과 화면 표시 이름 (dict 는 넣은 순서를 유지한다 → 요약 표시 순서가 된다)
FIELD_LABELS = {
    "visitor_name": "방문자",
    "phone": "휴대폰",
    "company": "소속",
    "visit_datetime": "방문일시",
    "host": "피방문자",
    "purpose": "방문목적",
    "vehicle_number": "차량번호",
}
REQUIRED_FIELDS = ["visitor_name", "phone", "company", "visit_datetime", "host", "purpose"]  # 차량번호는 선택

PHONE_FORMAT = re.compile(r"^(01[016789])-?(\d{3,4})-?(\d{4})$")  # 휴대폰번호 전체 형식 (^ 시작, $ 끝)
WEEKDAYS = "월화수목금토일"  # date.weekday() 는 월요일=0 ... 일요일=6 이다. WEEKDAYS[0] == "월"


def reference_date() -> date:
    """날짜 검증 기준일(오늘)을 돌려준다.

    환경변수 VISIT_REFERENCE_DATE(YYYY-MM-DD)가 있으면 그 날짜, 없으면 실제 KST 오늘이다.
    평가는 이 환경변수로 기준일을 고정해 언제 돌려도 같은 결과가 나오게 한다. API 요청으로는 바꿀 수 없다.
    """
    # os.getenv(이름) 은 System.getenv(이름) 과 같다. 값이 없으면 None.
    fixed = os.getenv("VISIT_REFERENCE_DATE")
    if fixed:
        return date.fromisoformat(fixed)
    return datetime.now(KST).date()


def normalize_phone(phone: str) -> str | None:
    """휴대폰번호를 010-1234-5678 형식으로 맞춘다. 형식이 틀리면 None 을 돌려준다."""
    # 공백과 점은 빼고 검사한다. `phone or ""` 는 None 이 들어와도 빈 문자열로 처리하기 위함이다.
    m = PHONE_FORMAT.match((phone or "").replace(" ", "").replace(".", ""))
    # 조건식 결과를 그대로 돌려준다: 일치하면 형식을 맞춘 문자열, 아니면 None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def format_visit_datetime(value: str) -> str:
    """'2026-11-17T10:00' 을 '2026-11-17 (화) 10:00' 처럼 요일을 넣어 읽기 쉽게 바꾼다. 해석이 안 되면 원문 그대로.

    시간대가 붙은 값('2026-11-16T14:00:00+09:00', 시드 신청 형식)도 같은 모양으로 바꾼다.
    """
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):  # 여러 예외를 튜플로 한 번에 잡는다 (Java 의 catch (A | B e))
        return value
    return f"{dt:%Y-%m-%d} ({WEEKDAYS[dt.weekday()]}) {dt:%H:%M}"


def _load_json(path: Path) -> list[dict]:
    """JSON 배열 파일을 읽는다. 파일이 없으면 빈 목록을 돌려준다."""
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def find_employee(name: str) -> dict | None:
    """임직원 명단에서 이름이 정확히 같은 사람을 찾는다. '박지훈 님' 처럼 붙은 호칭은 떼고 비교한다."""
    clean = (name or "").replace("님", "").strip()
    # next(제너레이터, 기본값): 조건에 맞는 첫 값, 없으면 기본값 (Java 의 stream().filter(...).findFirst().orElse(null))
    return next((e for e in _load_json(EMPLOYEES_PATH) if e["name"] == clean), None)


def validate_request(fields: dict) -> list[str]:
    """신청 필드를 검증해 오류 문장 목록을 돌려준다. 빈 목록이면 통과다.

    검증 규칙(CLAUDE.md 도메인 규칙): 필수 항목, 휴대폰번호 형식, 과거·당일 불가(최소 1일 전), 주말 불가, 피방문자 존재.
    """
    errors = []
    missing = [FIELD_LABELS[k] for k in REQUIRED_FIELDS if not fields.get(k)]
    if missing:
        errors.append(f"다음 정보가 더 필요합니다: {', '.join(missing)}")
    if fields.get("phone") and normalize_phone(fields["phone"]) is None:
        # 예시를 숫자로 쓰면(010-0000-0000) 응답 마스킹이 휴대폰번호로 보고 가려 버리므로 X 로 쓴다
        errors.append("휴대폰번호 형식이 올바르지 않습니다. 010-XXXX-XXXX 처럼 11자리로 알려 주세요.")
    if fields.get("visit_datetime"):
        try:
            visit = datetime.fromisoformat(fields["visit_datetime"]).date()
        except ValueError:  # 날짜 문자열을 해석하지 못한 경우 (Java 의 DateTimeParseException)
            errors.append("방문일시를 이해하지 못했습니다. 예: 2026년 11월 17일 오전 10시")
        else:  # try 블록에서 예외가 나지 않았을 때만 실행된다
            today = reference_date()
            if visit < today:
                errors.append("이미 지난 날짜로는 신청할 수 없습니다.")
            elif visit == today:
                errors.append("방문 당일 신청은 받지 않습니다. 방문일 최소 1일 전까지 신청해 주세요.")
            if visit.weekday() >= 5:  # 5=토요일, 6=일요일
                errors.append(f"주말({WEEKDAYS[visit.weekday()]}요일) 방문은 받지 않습니다. 평일로 다시 정해 주세요.")
    if fields.get("host") and find_employee(fields["host"]) is None:
        errors.append(f"피방문자 '{fields['host']}' 님을 임직원 명단에서 찾을 수 없습니다. 이름을 다시 확인해 주세요.")
    return errors


# 저장소 파일 잠금 (프로세스 안 공용). 서로 다른 대화가 동시에 확정해도 "읽고 → 번호 정하고 → 쓰기" 가 한 번에 하나씩만 실행된다.
# 잠금이 없으면 두 요청이 같은 파일 내용을 읽어 같은 신청번호를 발급하거나, 나중에 쓴 쪽이 먼저 쓴 기록을 덮어써 한 건이 사라진다.
# (uvicorn 워커 1개 = 프로세스 1개 전제. 여러 프로세스로 띄우면 파일 잠금이나 DB 가 필요하다)
_store_lock = threading.Lock()


def save_request(fields: dict) -> str:
    """검증된 신청을 저장소 JSON 에 추가하고 새 신청번호를 돌려준다.

    신청번호 형식: GM-기준일(YYYYMMDD)-일련번호 4자리. 시드 신청과 저장소를 함께 보고 같은 날짜 번호 다음 번호를 쓴다.
    원본 개인정보는 이 저장소에만 둔다(응답·trace·로그에는 마스킹된 값만 나간다).
    읽기부터 쓰기까지 전체를 저장소 잠금 안에서 실행해 번호 중복과 기록 유실을 막는다.
    """
    # with 잠금: 블록에 들어갈 때 잠그고, 블록을 나갈 때(예외가 나도) 자동으로 푼다.
    # Java 의 synchronized(lock) { ... } 또는 lock.lock(); try { ... } finally { lock.unlock(); } 와 같다.
    with _store_lock:
        return _append_request(fields)


def _append_request(fields: dict) -> str:
    """save_request 의 본체. 반드시 _store_lock 을 잡은 상태에서만 호출한다."""
    stored = _load_json(REQUESTS_PATH)
    prefix = f"GM-{reference_date():%Y%m%d}-"  # f-string 안의 :%Y%m%d 는 날짜 서식 지정 (Java 의 DateTimeFormatter)
    # set 컴프리헨션 {..}: 시드 신청을 취소하면 런타임에도 같은 번호 기록이 생기므로 같은 번호는 한 번만 센다 (Java 의 Set)
    used = {r["request_id"] for r in stored + _load_json(SEED_REQUESTS_PATH) if r["request_id"].startswith(prefix)}
    request_id = f"{prefix}{len(used) + 1:04d}"  # :04d 는 4자리 0 채우기 (Java 의 String.format("%04d", n))
    host = find_employee(fields["host"])
    record = {
        "request_id": request_id,
        "visitor_name": fields["visitor_name"],
        "phone": normalize_phone(fields["phone"]),
        "company": fields["company"],
        "visit_datetime": fields["visit_datetime"],
        "host": host["name"],
        "purpose": fields["purpose"],
        "vehicle_number": fields.get("vehicle_number"),
        "status": "submitted",
        "created_at": datetime.now(KST).isoformat(timespec="seconds"),
    }
    REQUESTS_PATH.parent.mkdir(parents=True, exist_ok=True)  # data/runtime 폴더가 없으면 만든다 (Files.createDirectories)
    REQUESTS_PATH.write_text(json.dumps(stored + [record], ensure_ascii=False, indent=2), encoding="utf-8")
    return request_id


# ══════════════════════════════════════════════════════════════════
# 도구
# ══════════════════════════════════════════════════════════════════


# response_format="content_and_artifact": 함수가 (content, artifact) 두 값을 돌려준다는 뜻이다.
#   - content: LLM 에게 보내는 텍스트 (검색된 문단)
#   - artifact: LLM 에게는 보내지 않고 프로그램 쪽에만 넘기는 데이터 (응답의 contexts 로 쓸 [{doc_id, text}] 목록)
# 반환 타입 tuple[str, list[dict[str, str]]] 은 Java 로 치면 Pair<String, List<Map<String, String>>> 다.
@tool(response_format="content_and_artifact")
def search_regulations(query: str) -> tuple[str, list[dict[str, str]]]:
    """사업장 방문 규정 문서를 검색한다. 방문 절차, 준비물, 반입 물품, 보안 서약, 주차, 개인정보 처리 질문에 쓴다.

    Args:
        query: 검색할 질문 또는 키워드
    """
    contexts = retriever.search(query)
    # 조각마다 "(doc_id=REG-03)" 머리말을 붙여, LLM 이 답변 끝에 근거 문서 번호를 적을 수 있게 한다.
    # "\n\n".join(목록) 은 목록을 빈 줄로 이어 붙인 문자열이다 (Java 의 String.join("\n\n", list)).
    content = "\n\n".join(f"(doc_id={c['doc_id']})\n{c['text']}" for c in contexts)
    return content, contexts


@tool
def find_host(name: str) -> str:
    """피방문자(방문객을 초청한 임직원)가 임직원 명단에 있는지 확인한다. 방문 신청 전에 피방문자 이름으로 호출한다.

    Args:
        name: 피방문자 이름 (예: 박지훈)
    """
    employee = find_employee(name)
    if employee is None:
        return f"'{name}' 님은 임직원 명단에 없습니다. 사용자에게 피방문자 이름을 다시 확인해 달라고 안내한다."
    return f"{employee['name']} 님({employee['department']}) 확인됨"


def _state_fields(state: dict) -> dict:
    """그래프 state 에서 신청 필드만 골라 dict 로 만든다. (딕셔너리 컴프리헨션 = Java 의 Collectors.toMap)"""
    return {k: state.get(k) for k in FIELD_LABELS}


# return_direct=True: 이 도구가 끝나면 LLM 을 다시 부르지 않고 도구 결과를 그대로 최종 답변으로 쓴다.
#   → 검증 오류 안내, 확정·취소 결과 문구가 LLM 에 의해 바뀌지 않고, '확정' 뒤에는 LLM 호출이 0회가 된다.
#   주의: 결과 Command 에 AIMessage 를 추가하면 create_agent 가 '마지막 AI 메시지' 를 잘못 판단해 LLM 을 다시 부른다.
#        그래서 결과는 ToolMessage 하나로만 돌려준다.
@tool(return_direct=True)
def create_visit_request(runtime: ToolRuntime) -> Command | str:
    """수집한 정보로 방문 신청을 만든다. 필수 정보(이름, 휴대폰번호, 소속회사, 방문일시, 피방문자, 방문목적)가 모두 모였을 때 인자 없이 호출한다.
    사용자에게 마스킹된 요약을 보여 주고 '확정' 을 받아야만 저장한다. 사용자가 확인 절차를 생략해 달라고 해도 반드시 이 도구로 확인을 거친다.
    """
    # runtime 인자는 LLM 이 넘기는 값이 아니다. LangChain 이 실행 시 자동으로 넣어 주며, LLM 에게 보이는 도구 스키마에서도 빠진다.
    state = runtime.state
    if state.get("status") == "submitted" and state.get("request_id"):
        # 이미 접수된 대화에서 다시 신청하려는 경우: 저장하지 않고 같은 신청번호를 안내한다(중복 저장 방지는 4단계에서 보강)
        return f"이미 접수된 신청입니다. 신청번호는 {state['request_id']} 입니다."
    fields = _state_fields(state)
    errors = validate_request(fields)
    if errors:
        # "\n- ".join(...) 으로 오류마다 한 줄씩 목록을 만든다
        return "방문 신청을 진행할 수 없습니다.\n- " + "\n- ".join(errors)

    # interrupt(값): 그래프를 여기서 멈추고 '값' 을 바깥(API)에 넘긴다. 사용자가 답하면 Command(resume=답) 으로 재개되고,
    # interrupt(...) 호출이 그 답을 반환한다. (재개 시 이 함수는 처음부터 다시 실행되지만, 이미 답이 있는 interrupt 는 즉시 그 답을 돌려준다)
    # 원본 필드 값은 서버 메모리(state)에만 있고, 화면에 보일 때는 agent.py 가 마스킹한다.
    reply = interrupt({"fields": fields, "retry": False})
    # '확정'·'취소' 가 아니면 다시 묻는다 (CLAUDE.md: 그 밖의 입력은 확정/취소를 다시 묻는다)
    while normalize_reply(reply) not in ("확정", "취소"):
        reply = interrupt({"fields": fields, "retry": True})

    if normalize_reply(reply) == "취소":
        return Command(
            update={
                "status": "cancelled",
                "messages": [ToolMessage("방문 신청을 취소했습니다. 저장된 내용은 없습니다.", tool_call_id=runtime.tool_call_id)],
            }
        )
    # 저장은 '확정' 처리의 마지막 단계다. 저장과 state 갱신(status, request_id)을 한 Command 로 함께 반영한다.
    request_id = save_request(fields)
    message = (
        f"방문 신청이 접수되었습니다. 신청번호는 {request_id} 입니다. "
        "방문 당일 정문 안내데스크에서 신청번호와 신분증을 보여 주세요."
    )
    return Command(
        update={
            "status": "submitted",
            "request_id": request_id,
            "messages": [ToolMessage(message, tool_call_id=runtime.tool_call_id)],
        }
    )


# ══════════════════════════════════════════════════════════════════
# 5단계: 본인 신청 목록·상세 조회와 취소
# ══════════════════════════════════════════════════════════════════

# 본인 조회 실패 문구. 이름·휴대폰번호·신청번호 중 무엇이 틀렸든, 신청이 0건이든 항상 이 한 문장만 돌려준다.
# 틀린 항목이나 신청 존재 여부를 알려 주면 남의 신청을 추측해 맞춰 보는 데 쓰일 수 있기 때문이다 (CLAUDE.md 도메인 규칙).
LOOKUP_NOT_FOUND = "입력하신 정보와 일치하는 신청을 찾을 수 없습니다. 이름, 휴대폰번호, 신청번호를 다시 확인해 주세요."
LOOKUP_NEEDS_BOTH = "본인 신청 조회에는 이름과 휴대폰번호가 함께 필요합니다. 두 가지를 같이 알려 주세요."
LOOKUP_OTHER_PERSON = (
    "이 대화에서는 처음 확인한 본인의 신청만 조회할 수 있습니다. 다른 이름이나 휴대폰번호로 조회하려면 새 대화를 시작해 주세요."
)
LOOKUP_NEEDS_LIST = "먼저 이름과 휴대폰번호로 본인 신청 목록을 조회해 주세요. 목록의 번호로 상세를 볼 수 있습니다."
CANCEL_NEEDS_FULL_ID = (
    "취소하려면 신청번호 전체(예: GM-20261110-0001 형식)를 입력해 주세요. 화면에는 일부만 표시되므로 접수 때 받은 번호를 확인해 주세요."
)
FULL_REQUEST_ID = re.compile(r"GM-\d{8}-\d{4}")  # 가리지 않은 전체 신청번호 형식


def mask_request_id(request_id: str) -> str:
    """신청번호를 일부만 보이게 가린다. 'GM-20261110-0001' → 'GM-20261110-**01'.

    목록·상세에서는 이 형태만 보여 준다. 취소는 전체 번호를 알아야 하므로, 화면을 본 사람이 번호를 그대로 옮겨 취소할 수 없다.
    """
    # rpartition("-"): 마지막 '-' 기준으로 (앞, '-', 뒤) 세 조각으로 나눈다. 언패킹으로 세 변수에 한 번에 받는다
    # (Java 라면 lastIndexOf("-") 로 substring 두 번). _ 는 쓰지 않는 값이라는 관례적 이름이다.
    head, _, tail = request_id.rpartition("-")
    return f"{head}-**{tail[-2:]}"  # tail[-2:] 는 끝 2글자 (Java 의 tail.substring(tail.length() - 2))


def _visit_date(record: dict) -> date:
    """신청 기록의 방문 날짜(date)."""
    return datetime.fromisoformat(record["visit_datetime"]).date()


def is_cancellable(record: dict) -> bool:
    """취소 가능 여부: 접수 상태(submitted)이고 방문일이 기준일(KST 오늘)보다 뒤일 때만. 당일·지난 방문은 취소할 수 없다."""
    return record.get("status") == "submitted" and _visit_date(record) > reference_date()


def _status_label(record: dict) -> str:
    """신청 상태를 화면 문구로 바꾼다. 지난 방문은 저장된 status 가 submitted 여도 '지난 방문' 으로 보여 준다."""
    if record.get("status") == "cancelled":
        return "취소됨"
    if _visit_date(record) < reference_date():
        return "지난 방문"
    return "접수 완료" if is_cancellable(record) else "접수 완료 (당일 방문, 취소 불가)"


def _merged_records() -> list[dict]:
    """시드 신청과 런타임 저장소를 신청번호 기준으로 합친다. 같은 번호면 런타임 기록이 이긴다.

    시드 파일(data/seed_requests.json)은 커밋된 읽기 전용 파일이라, 시드 신청을 취소하면 런타임 저장소에 취소 기록을 따로 남긴다.
    반드시 _store_lock 을 잡은 상태에서만 호출한다.
    """
    # 딕셔너리 컴프리헨션으로 {신청번호: 기록} Map 을 만들고, update 로 런타임 기록을 덮어쓴다 (Java 의 map.putAll)
    merged = {r["request_id"]: r for r in _load_json(SEED_REQUESTS_PATH)}
    merged.update({r["request_id"]: r for r in _load_json(REQUESTS_PATH)})
    return list(merged.values())


def _is_owner(record: dict, name: str, phone: str) -> bool:
    """기록의 방문자 이름과 휴대폰번호가 둘 다 같은지. 휴대폰번호는 하이픈 유무와 상관없이 비교한다."""
    phone_norm = normalize_phone(phone)
    return (
        phone_norm is not None
        and record["visitor_name"] == (name or "").strip()
        and normalize_phone(record["phone"]) == phone_norm
    )


def find_requests(name: str, phone: str) -> list[dict]:
    """이름+휴대폰번호가 함께 맞는 본인 신청 전체를 방문일 순으로 돌려준다(지난 방문·취소 건 포함). 없으면 빈 목록."""
    with _store_lock:  # 저장소 쓰기(저장·취소)와 겹치지 않게 잠금 안에서 읽는다
        records = _merged_records()
    # sorted(목록, key=함수): 함수 결과 기준 정렬 (Java 의 list.sort(Comparator.comparing(...)))
    # lambda r: ... 는 이름 없는 한 줄 함수 (Java 의 r -> ...)
    return sorted((r for r in records if _is_owner(r, name, phone)), key=lambda r: r["visit_datetime"])


def cancel_request(name: str, phone: str, request_id: str) -> tuple[str, dict | None]:
    """본인 신청 한 건을 취소한다. 삭제하지 않고 status=cancelled, cancelled_at 을 남긴다.

    읽기부터 쓰기까지 저장소 잠금 안에서 실행해, 동시에 들어온 저장·취소가 서로의 기록을 덮어쓰지 않게 한다.

    Args:
        name: 방문자 본인 이름
        phone: 휴대폰번호
        request_id: 신청번호 전체 (대소문자·앞뒤 공백 무시)

    Returns:
        (결과 코드, 기록) 튜플. 결과 코드는 not_found / already / too_late / cancelled 중 하나다.
        not_found 면 기록은 None 이다. (Java 라면 결과 enum 과 기록을 담은 작은 record 클래스를 돌려줄 자리)
    """
    rid = (request_id or "").strip().upper()
    with _store_lock:
        record = next((r for r in _merged_records() if r["request_id"] == rid and _is_owner(r, name, phone)), None)
        if record is None:
            return "not_found", None
        if record.get("status") == "cancelled":
            return "already", record
        if not is_cancellable(record):
            return "too_late", record
        # {**기록, 키: 값}: 기존 dict 를 복사하면서 일부 키만 바꾼 새 dict (Java 라면 new HashMap<>(record) 후 put)
        updated = {**record, "status": "cancelled", "cancelled_at": datetime.now(KST).isoformat(timespec="seconds")}
        stored = [r for r in _load_json(REQUESTS_PATH) if r["request_id"] != rid]  # 런타임에 같은 번호가 있으면 교체
        REQUESTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        REQUESTS_PATH.write_text(json.dumps(stored + [updated], ensure_ascii=False, indent=2), encoding="utf-8")
        return "cancelled", updated


def _identity(state: dict, name: str, phone: str) -> tuple[str | None, str | None, str | None]:
    """이번 조회·취소에 쓸 본인(이름, 휴대폰번호)을 정한다.

    - 인자로 이름·휴대폰번호가 오면 그 값을 쓴다. 단, 이 대화에서 이미 확인된 본인과 다르면 거절한다(처음 확인된 본인만 조회).
    - 인자가 없으면 이 대화에서 이미 확인된 본인을 쓴다("다시 보여 줘", "1번 취소" 같은 이어지는 요청).

    Returns:
        (이름, 휴대폰번호, 거절 문구). 거절이면 앞 두 값은 None, 아니면 거절 문구가 None 이다.
    """
    bound_name, bound_phone = state.get("lookup_name"), state.get("lookup_phone")
    if not (name or phone):
        if bound_name:
            return bound_name, bound_phone, None
        return None, None, LOOKUP_NEEDS_BOTH
    if not (name and phone):
        return None, None, LOOKUP_NEEDS_BOTH
    if bound_name and (name.strip(), normalize_phone(phone)) != (bound_name, bound_phone):
        return None, None, LOOKUP_OTHER_PERSON
    return name.strip(), phone, None


def _reply(runtime: ToolRuntime, text: str, update: dict | None = None, artifact: dict | None = None) -> Command:
    """return_direct 도구의 결과를 ToolMessage 하나만 담은 Command 로 만든다(필요하면 state 갱신도 함께).

    결과에 AIMessage 를 넣으면 create_agent 가 LLM 을 다시 부르므로 ToolMessage 하나만 넣는다(create_visit_request 와 같은 이유).

    Args:
        runtime: 도구 실행 정보 (tool_call_id 를 ToolMessage 에 연결한다)
        text: 사용자에게 그대로 보일 결과 문장
        update: 함께 바꿀 state 값 (없으면 메시지만 추가)
        artifact: LLM 에게 보내지 않고 agent.py 에만 넘기는 표시값 (취소 성공 여부)
    """
    message = ToolMessage(text, tool_call_id=runtime.tool_call_id, artifact=artifact)
    # **dict: dict 를 풀어 키·값을 그대로 넣는다. (update or {}) 는 update 가 None 이면 빈 dict 를 쓰라는 뜻
    return Command(update={**(update or {}), "messages": [message]})


@tool(return_direct=True)
def get_my_requests(runtime: ToolRuntime, name: str = "", phone: str = "") -> Command:
    """본인의 방문 신청 목록을 조회한다. 방문자 본인의 이름과 휴대폰번호를 함께 받아 호출한다.
    이 대화에서 이미 본인이 확인됐으면 인자 없이 호출해도 된다. 다른 사람의 신청이나 전체 목록은 조회할 수 없다.

    Args:
        name: 신청한 방문자 본인 이름
        phone: 신청할 때 쓴 휴대폰번호
    """
    # return_direct=True: 결과를 LLM 이 다시 쓰지 않고 그대로 답한다 → 실패 문구가 항상 똑같이 나간다.
    # 결과의 이름·휴대폰번호는 이 도구가 아니라 agent.py 가 응답으로 내보내기 전에 마스킹한다.
    who, who_phone, refusal = _identity(runtime.state, name, phone)
    if refusal:
        return _reply(runtime, refusal)
    records = find_requests(who, who_phone)
    if not records:
        return _reply(runtime, LOOKUP_NOT_FOUND)
    lines = [f"본인 신청 {len(records)}건을 확인했습니다. (방문일 순, 신청번호는 일부만 표시)"]
    # enumerate(목록, 1): (번호, 값) 쌍을 1번부터 돌려준다 (Java 의 for (int i = 1; ...) 와 get(i - 1))
    for number, r in enumerate(records, 1):
        cancel_note = " | 취소 가능" if is_cancellable(r) else ""
        lines.append(
            f"{number}. {mask_request_id(r['request_id'])} | {format_visit_datetime(r['visit_datetime'])}"
            f" | 피방문자 {r['host']} | {_status_label(r)}{cancel_note}"
        )
    lines.append("번호를 말씀하시면 상세를 보여 드립니다. 취소하려면 신청번호 전체를 알려 주세요.")
    # 처음 확인된 본인을 state 에 묶어 둔다(이후 다른 본인 조회 거절, 상세·취소에서 재사용).
    # lookup_ids 는 목록 순서대로의 전체 신청번호다. 화면에는 나가지 않고 'N번 상세' 를 풀 때만 쓴다.
    update = {
        "lookup_name": who,
        "lookup_phone": normalize_phone(who_phone),
        "lookup_ids": [r["request_id"] for r in records],
    }
    return _reply(runtime, "\n".join(lines), update)


@tool(return_direct=True)
def get_request_detail(runtime: ToolRuntime, number: int) -> Command:
    """직전에 조회한 본인 신청 목록에서 고른 신청의 상세를 보여 준다. '2번 자세히' 처럼 목록 번호로 호출한다.

    Args:
        number: 목록의 번호 (1부터)
    """
    ids = runtime.state.get("lookup_ids") or []
    if not ids:
        return _reply(runtime, LOOKUP_NEEDS_LIST)
    if not 1 <= number <= len(ids):  # 파이썬은 범위 비교를 이어 쓸 수 있다 (Java 의 1 <= n && n <= size)
        return _reply(runtime, f"목록 번호는 1부터 {len(ids)} 사이로 골라 주세요.")
    # 목록을 본 뒤 취소됐을 수 있으므로 저장소를 다시 읽는다. ids[number - 1]: 1번 → 인덱스 0
    records = find_requests(runtime.state["lookup_name"], runtime.state["lookup_phone"])
    record = next((r for r in records if r["request_id"] == ids[number - 1]), None)
    if record is None:
        return _reply(runtime, LOOKUP_NOT_FOUND)
    lines = [f"{number}번 신청 상세입니다.", f"신청번호: {mask_request_id(record['request_id'])}"]
    for key, label in FIELD_LABELS.items():
        value = record.get(key)
        if key == "visit_datetime" and value:
            value = format_visit_datetime(value)
        lines.append(f"{label}: {value or '없음'}")
    lines.append(f"상태: {_status_label(record)}")
    if record.get("cancelled_at"):
        lines.append(f"취소일시: {format_visit_datetime(record['cancelled_at'])}")
    if is_cancellable(record):
        lines.append("이 신청을 취소하려면 신청번호 전체를 알려 주세요.")
    return _reply(runtime, "\n".join(lines))


@tool(return_direct=True)
def cancel_my_request(runtime: ToolRuntime, request_id: str = "", name: str = "", phone: str = "") -> Command:
    """본인의 방문 신청을 취소한다. 사용자가 직접 입력한 신청번호 전체(GM-YYYYMMDD-NNNN)를 그대로 넣어 호출한다.
    사용자가 전체 번호를 말하지 않았으면 request_id 를 비워 호출한다(도구가 전체 번호를 요청한다). 확인 질문 없이 바로 취소된다.
    이 대화에서 본인이 아직 확인되지 않았으면 이름과 휴대폰번호도 함께 넣는다.

    Args:
        request_id: 취소할 신청번호 전체
        name: 신청한 방문자 본인 이름 (이 대화에서 이미 확인됐으면 생략)
        phone: 신청할 때 쓴 휴대폰번호 (이 대화에서 이미 확인됐으면 생략)
    """
    who, who_phone, refusal = _identity(runtime.state, name, phone)
    if refusal:
        return _reply(runtime, refusal)
    # 일부만 가린 번호(GM-20261110-**01)나 빈 값이면 취소하지 않고 전체 번호를 요청한다
    if not FULL_REQUEST_ID.fullmatch((request_id or "").strip().upper()):
        return _reply(runtime, CANCEL_NEEDS_FULL_ID)
    outcome, record = cancel_request(who, who_phone, request_id)  # 튜플 언패킹: 두 반환값을 두 변수로
    if outcome == "not_found":
        return _reply(runtime, LOOKUP_NOT_FOUND)
    masked = mask_request_id(record["request_id"])
    when = format_visit_datetime(record["visit_datetime"])
    if outcome == "already":
        return _reply(runtime, f"이미 취소된 신청입니다. ({masked}, 방문일시 {when})")
    if outcome == "too_late":
        return _reply(
            runtime, f"방문 당일이거나 지난 방문은 취소할 수 없습니다. ({masked}, 방문일시 {when}) 피방문자에게 문의해 주세요."
        )
    # 첫 확인이 취소로 이뤄진 경우에도 본인을 state 에 묶어 둔다. artifact 의 cancelled 표시는 agent.py 가 이 응답의 status 를
    # cancelled 로 돌려주는 데 쓴다(대화 state 의 status 는 바꾸지 않는다: 같은 대화의 방문 신청 진행 상태와 섞이지 않게).
    update = {"lookup_name": who, "lookup_phone": normalize_phone(who_phone)}
    text = f"신청을 취소했습니다. ({masked}, 방문일시 {when}, 피방문자 {record['host']})"
    return _reply(runtime, text, update, artifact={"cancelled": True})


def normalize_reply(reply) -> str:
    """'확정'·'취소' 답을 비교용으로 다듬는다. 앞뒤 공백과 끝의 마침표·느낌표를 지운다. (agent.py 의 중복 확정 판별에서도 쓴다)"""
    return str(reply).strip().rstrip(".!")
