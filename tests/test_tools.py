"""본인 신청 목록·상세 조회와 취소 테스트 (AC4). 5단계: get_my_requests, get_request_detail, cancel_my_request.

규칙(CLAUDE.md 도메인 규칙):
- 목록·상세는 이름+휴대폰번호가 함께 맞을 때만 보여 준다(지난 방문·취소 건 포함, 방문일 순, 신청번호는 일부만 표시).
- 불일치·0건은 신청이 있는지 없는지 드러내지 않는 같은 문구로 답한다.
- 한 대화에서는 처음 확인된 본인만 조회한다.
- 취소는 전체 신청번호까지 맞고 방문일이 기준일(2026-11-12)보다 뒤인 접수 건만 즉시 처리한다. 삭제 대신 status=cancelled.
시드 신청(data/seed_requests.json): GM-20261110-0001, 김방문, 010-2222-3333, 2026-11-16(월) 14:00

[Java 개발자를 위한 개요]
- 도구가 대화 state(처음 확인된 본인, 마지막 목록)를 쓰므로, 도구를 직접 부르지 않고 /query API 를 여러 턴 호출해 확인한다.
- 가짜 LLM(ScriptedChatModel)은 턴마다 정해 둔 도구 호출을 하나씩 돌려준다. 세 도구 모두 return_direct 라서
  한 턴에 LLM 호출은 1번이고, 도구 결과 문장이 그대로 answer 가 된다.
"""

import json
import re

from fastapi.testclient import TestClient

from src import tools
from src.api import app
from tests.conftest import tool_call

client = TestClient(app)

ME = {"name": "김방문", "phone": "010-2222-3333"}
SEED_ID = "GM-20261110-0001"
RAW_PHONE = re.compile(r"01[016789]-?\d{3,4}-?\d{4}")


def _record(request_id: str, visit: str, status: str = "submitted", name: str = "김방문", phone: str = "010-2222-3333") -> dict:
    """런타임 저장소에 넣을 가상 신청 한 건 (visit 예: '2026-11-05T10:00')."""
    return {
        "request_id": request_id,
        "visitor_name": name,
        "phone": phone,
        "company": "가상테크",
        "visit_datetime": visit,
        "host": "박지훈",
        "purpose": "설비 점검 협의",
        "vehicle_number": None,
        "status": status,
        "created_at": "2026-11-01T09:00:00+09:00",
    }


def _put(store, *records):
    """임시 저장소 파일에 신청 기록을 쓴다. *records 는 가변 인자 (Java 의 Record... records)."""
    store.write_text(json.dumps(list(records), ensure_ascii=False), encoding="utf-8")


def _chat(use_model, turns: list[tuple[str, dict]]) -> list[dict]:
    """한 대화(thread)에서 (도구 이름, 인자) 순서대로 도구를 부르게 하고, 턴마다의 응답 body 목록을 돌려준다."""
    # 리스트 컴프리헨션으로 턴마다 다른 호출 ID 의 도구 호출 응답을 만든다
    use_model([tool_call(name, args, f"call-{i}") for i, (name, args) in enumerate(turns)])
    bodies, thread_id = [], None
    for name, _ in turns:
        payload = {"question": f"{name} 요청"} | ({"thread_id": thread_id} if thread_id else {})  # | 는 dict 합치기
        body = client.post("/query", json=payload).json()
        thread_id = body["thread_id"]
        bodies.append(body)
    return bodies


# ── 목록 ──────────────────────────────────────────────────────────


def test_이름과_휴대폰번호로_본인_신청_목록을_보여준다(use_model):
    """시나리오 8: 시드 신청이 목록에 나온다. 신청번호는 일부만, 휴대폰번호·이름은 마스킹, 대화 status 는 그대로."""
    [body] = _chat(use_model, [("get_my_requests", ME)])  # 원소 1개짜리 리스트 언패킹
    assert body["answer"].startswith("본인 신청 1건을 확인했습니다.")
    assert "1. GM-20261110-**01 | 2026-11-16 (월) 14:00 | 피방문자 박지훈 | 접수 완료 | 취소 가능" in body["answer"]
    assert SEED_ID not in str(body)  # 전체 신청번호는 나가지 않는다
    assert "김방문" not in str(body)
    assert not RAW_PHONE.search(str(body))
    assert body["status"] == "collecting"


def test_휴대폰번호는_하이픈이_없어도_같은_번호로_본다(use_model):
    [body] = _chat(use_model, [("get_my_requests", {"name": "김방문", "phone": "01022223333"})])
    assert body["answer"].startswith("본인 신청 1건을 확인했습니다.")


def test_지난_방문과_취소_건도_방문일_순으로_보여준다(use_model, isolated_store):
    """지난 방문(11-05), 당일(11-12), 취소(11-20) 건이 시드(11-16)와 함께 방문일 순으로 나온다. 취소 가능 표시는 시드에만."""
    _put(
        isolated_store,
        _record("GM-20261120-0001", "2026-11-20T10:00", status="cancelled"),
        _record("GM-20261101-0001", "2026-11-05T10:00"),
        _record("GM-20261111-0001", "2026-11-12T15:00"),
    )
    [body] = _chat(use_model, [("get_my_requests", ME)])
    lines = body["answer"].splitlines()
    assert lines[0].startswith("본인 신청 4건")
    assert lines[1].startswith("1. GM-20261101-**01 | 2026-11-05") and lines[1].endswith("지난 방문")
    assert lines[2].startswith("2. GM-20261111-**01 | 2026-11-12") and "취소 불가" in lines[2]
    assert lines[3].startswith("3. GM-20261110-**01 | 2026-11-16") and lines[3].endswith("취소 가능")
    assert lines[4].startswith("4. GM-20261120-**01 | 2026-11-20") and lines[4].endswith("취소됨")


def test_불일치와_0건은_모두_같은_문구로_답한다(use_model):
    """시나리오 16(휴대폰번호 한 자리 틀림), 이름만 틀림, 아예 없는 사람이 모두 같은 문장이다(존재 여부 비노출)."""
    answers = set()  # 같은 문자열은 하나로 합쳐지는 Set
    for args in [
        {"name": "김방문", "phone": "010-2222-3334"},
        {"name": "김방운", "phone": "010-2222-3333"},
        {"name": "없는사람", "phone": "010-1111-0000"},
    ]:
        [body] = _chat(use_model, [("get_my_requests", args)])
        answers.add(body["answer"])
    assert answers == {tools.LOOKUP_NOT_FOUND}


def test_이름이나_휴대폰번호_하나만으로는_조회하지_않는다(use_model):
    [body] = _chat(use_model, [("get_my_requests", {"name": "김방문"})])
    assert body["answer"] == tools.LOOKUP_NEEDS_BOTH


def test_한_대화에서_다른_본인으로는_조회하지_않는다(use_model, isolated_store):
    """처음 확인된 본인(김방문) 뒤에 다른 사람(홍길동)으로 조회하면 새 대화로 안내한다. 다른 사람의 신청은 나오지 않는다."""
    _put(isolated_store, _record("GM-20261111-0002", "2026-11-17T10:00", name="홍길동", phone="010-1234-5678"))
    _, second = _chat(
        use_model, [("get_my_requests", ME), ("get_my_requests", {"name": "홍길동", "phone": "010-1234-5678"})]
    )
    assert second["answer"] == tools.LOOKUP_OTHER_PERSON
    assert "GM-20261111" not in second["answer"]


def test_새로_저장한_신청도_목록에_나온다():
    """런타임 저장소에 저장된 신청도 같은 규칙으로 찾는다 (저장 → find_requests)."""
    form = {
        "visitor_name": "홍길동",
        "phone": "010-1234-5678",
        "company": "한빛정밀",
        "visit_datetime": "2026-11-17T10:00",
        "host": "박지훈",
        "purpose": "설비 미팅",
    }
    request_id = tools.save_request(form)
    assert [r["request_id"] for r in tools.find_requests("홍길동", "01012345678")] == [request_id]


# ── 상세 ──────────────────────────────────────────────────────────


def test_목록_번호로_상세를_보여준다(use_model):
    """목록 뒤 '1번 자세히' → 상세. 인자 없이 와도 이 대화에서 확인된 본인 이름은 마스킹된다."""
    _, detail = _chat(use_model, [("get_my_requests", ME), ("get_request_detail", {"number": 1})])
    text = detail["answer"]
    assert text.startswith("1번 신청 상세입니다.")
    assert "신청번호: GM-20261110-**01" in text
    assert "방문자: 김*문" in text
    assert "방문목적: 설비 점검 협의" in text
    assert "상태: 접수 완료" in text
    assert SEED_ID not in text and not RAW_PHONE.search(text)


def test_목록_없이_상세를_요청하면_본인_확인부터_안내한다(use_model):
    [body] = _chat(use_model, [("get_request_detail", {"number": 1})])
    assert body["answer"] == tools.LOOKUP_NEEDS_LIST


# ── 취소 ──────────────────────────────────────────────────────────


def test_전체_신청번호로_즉시_취소하고_기록을_남긴다(use_model, isolated_store):
    """시드 신청 취소: 응답 status=cancelled, 런타임 저장소에 cancelled 기록, 시드 파일은 그대로, 다시 조회하면 '취소됨'."""
    seed_before = tools.SEED_REQUESTS_PATH.read_text(encoding="utf-8")
    _, cancel, again = _chat(
        use_model,
        [("get_my_requests", ME), ("cancel_my_request", {"request_id": SEED_ID}), ("get_my_requests", {})],
    )
    assert cancel["answer"].startswith("신청을 취소했습니다. (GM-20261110-**01")
    assert cancel["status"] == "cancelled"
    [stored] = json.loads(isolated_store.read_text(encoding="utf-8"))
    assert stored["request_id"] == SEED_ID and stored["status"] == "cancelled" and stored["cancelled_at"]
    assert tools.SEED_REQUESTS_PATH.read_text(encoding="utf-8") == seed_before
    assert again["answer"].splitlines()[1].endswith("취소됨")
    assert again["status"] == "collecting"  # 취소 성공 응답만 cancelled, 이후 조회는 대화 status 그대로


def test_이름과_휴대폰번호를_함께_주면_목록_없이도_취소한다(use_model):
    [body] = _chat(use_model, [("cancel_my_request", {"request_id": SEED_ID.lower(), **ME})])
    assert body["status"] == "cancelled"


def test_이미_취소된_신청은_다시_바꾸지_않는다(use_model, isolated_store):
    _put(isolated_store, _record(SEED_ID, "2026-11-16T14:00:00+09:00", status="cancelled") | {"cancelled_at": "X"})
    [body] = _chat(use_model, [("cancel_my_request", {"request_id": SEED_ID, **ME})])
    assert body["answer"].startswith("이미 취소된 신청입니다.")
    assert body["status"] == "collecting"
    assert json.loads(isolated_store.read_text(encoding="utf-8"))[0]["cancelled_at"] == "X"


def test_당일과_지난_방문은_취소하지_않는다(use_model, isolated_store):
    _put(isolated_store, _record("GM-20261111-0001", "2026-11-12T15:00"), _record("GM-20261101-0001", "2026-11-05T10:00"))
    before = isolated_store.read_text(encoding="utf-8")
    for rid in ["GM-20261111-0001", "GM-20261101-0001"]:
        [body] = _chat(use_model, [("cancel_my_request", {"request_id": rid, **ME})])
        assert body["answer"].startswith("방문 당일이거나 지난 방문은 취소할 수 없습니다.")
        assert body["status"] == "collecting"
    assert isolated_store.read_text(encoding="utf-8") == before


def test_번호가_틀리거나_일부만_주면_취소하지_않는다(use_model, isolated_store):
    """틀린 번호·남의 번호는 같은 불일치 문구, 가린 번호(**)·빈 번호는 전체 번호를 요청한다. 저장소는 그대로."""
    _put(isolated_store, _record("GM-20261111-0002", "2026-11-17T10:00", name="홍길동", phone="010-1234-5678"))
    before = isolated_store.read_text(encoding="utf-8")
    cases = {
        "GM-20261110-9999": tools.LOOKUP_NOT_FOUND,  # 없는 번호
        "GM-20261111-0002": tools.LOOKUP_NOT_FOUND,  # 다른 사람의 신청
        "GM-20261110-**01": tools.CANCEL_NEEDS_FULL_ID,  # 화면의 가린 번호
        "": tools.CANCEL_NEEDS_FULL_ID,  # 'N번 취소' 처럼 번호 없이
    }
    for rid, expected in cases.items():  # .items() 는 (키, 값) 쌍 (Java 의 entrySet())
        [body] = _chat(use_model, [("cancel_my_request", {"request_id": rid, **ME})])
        assert body["answer"] == expected
    assert isolated_store.read_text(encoding="utf-8") == before


def test_본인_확인_없이_N번_취소를_요청하면_이름과_휴대폰번호를_묻는다(use_model):
    [body] = _chat(use_model, [("cancel_my_request", {})])
    assert body["answer"] == tools.LOOKUP_NEEDS_BOTH
