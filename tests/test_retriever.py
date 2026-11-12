"""RAG 파이프라인 테스트 (AC1). 가짜 임베딩으로 Bedrock 없이 검증한다.

pytest 는 test_ 로 시작하는 함수를 모두 찾아 실행한다. 함수 이름을 한글로 써도 되므로, 이름만 읽어도 무엇을 검증하는지 알 수 있게 지었다.
인자에 fake_index 가 있는 테스트는 conftest.py 의 fake_index 픽스처가 먼저 실행되어 가짜 인덱스가 준비된다.
"""

from src import retriever
from tests.conftest import KeywordEmbeddings


def test_규정_문서는_5개에서_10개다():
    """data/regulations 의 문서 수가 Seed 가 정한 범위(5~10개) 안에 있다."""
    # 세트 컴프리헨션 {식 for ...} 은 중복 없는 Set 을 만든다 → 청크가 여러 개여도 문서 번호는 한 번만 센다
    doc_ids = {d.metadata["doc_id"] for d in retriever.load_documents()}
    # 파이썬은 비교 연산을 이어 쓸 수 있다: 5 <= x <= 10 은 (5 <= x && x <= 10)
    assert 5 <= len(doc_ids) <= 10


def test_모든_청크에_doc_id_와_title_이_있다():
    """청크마다 근거 표시에 쓸 doc_id 와 title 을 가진다."""
    for doc in retriever.load_documents():
        assert doc.metadata["doc_id"].startswith("REG-")
        assert doc.metadata["title"]  # 빈 문자열이 아니면 참 (파이썬은 빈 문자열을 False 로 취급)
        assert doc.page_content.strip()


def test_노트북_반입_질문은_반입물품_규정을_찾는다(fake_index):
    """'노트북 반입' 검색 결과에 REG-03(반입 물품 규정)이 들어 있다."""
    results = retriever.search("노트북을 가지고 들어가도 되나요? 노트북 반입")
    assert "REG-03" in [r["doc_id"] for r in results]  # `in` 은 리스트 포함 여부 (Java 의 list.contains)


def test_검색_결과는_doc_id_와_text_를_가진다(fake_index):
    """API 응답 contexts 형식 {doc_id, text} 로 돌려준다."""
    results = retriever.search("주차", k=2)
    assert len(results) == 2
    # all(...) 은 모든 원소가 참일 때만 참 (Java 의 stream().allMatch). set(dict) 는 dict 의 키 집합이다.
    assert all(set(r) == {"doc_id", "text"} for r in results)


def test_식당_내용은_규정_문서에_없다():
    """규정에 없는 주제(구내식당)는 문서 어디에도 없다. 에이전트가 이 주제로 답을 지어내지 않아야 하는 근거다."""
    # any(...) 는 하나라도 참이면 참 (Java 의 stream().anyMatch)
    assert not any("식당" in d.page_content for d in retriever.load_documents())


def test_인덱스는_가짜_임베딩으로도_만들어진다():
    """build_index 가 임베딩을 주입받는다(단위 테스트에서 Bedrock 을 부르지 않기 위한 구조 확인)."""
    index = retriever.build_index(KeywordEmbeddings())
    # index.index.ntotal 은 FAISS 에 들어간 벡터 개수다. 청크 수와 같아야 한다.
    assert index.index.ntotal == len(retriever.load_documents())
