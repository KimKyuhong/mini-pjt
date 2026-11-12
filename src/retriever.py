"""RAG 파이프라인: 가상 규정 문서를 섹션 단위로 나눠 FAISS 로 검색한다.

[Java 개발자를 위한 개요]
- RAG(Retrieval-Augmented Generation): LLM 이 답하기 전에 관련 문서를 먼저 찾아 근거로 붙여 주는 방식이다.
- 처리 순서
  1) data/regulations/*.md 파일을 읽어 '## ' 제목 단위 조각(청크)으로 자른다.
  2) 각 조각을 임베딩(문장 의미를 숫자 벡터로 바꾼 것)으로 변환한다. 여기서는 Amazon Titan 임베딩을 쓴다.
  3) 벡터를 FAISS(메모리 벡터 검색 라이브러리)에 넣어 둔다.
  4) 질문이 오면 질문도 벡터로 바꾸고, 가장 가까운 조각 k 개를 돌려준다.
- 인덱스는 서버 메모리에만 있다. 서버를 띄운 뒤 첫 검색 때 한 번 만든다(그래서 첫 질문만 몇 초 더 걸린다).
"""

from pathlib import Path  # 파일 경로 (java.nio.file.Path)

from botocore.config import Config  # AWS SDK 클라이언트 설정 (타임아웃, 재시도)
from langchain_aws import BedrockEmbeddings  # Bedrock 임베딩 클라이언트
from langchain_community.vectorstores import FAISS  # 벡터 검색 저장소
from langchain_core.documents import Document  # 문서 조각 객체 (본문 + 메타데이터)
from langchain_core.embeddings import Embeddings  # 임베딩 공통 인터페이스 (테스트에서 가짜 구현을 끼울 수 있다)

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
REGION = "us-east-1"
# 프로젝트 루트/data/regulations 폴더. Path 는 / 연산자로 하위 경로를 붙인다.
REGULATIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "regulations"

# Bedrock 호출 공통 설정: 30초 타임아웃, 재시도 없음 (CLAUDE.md '주고받는 형식' 규칙).
# total_max_attempts=1 은 "최초 1회만 시도" 라는 뜻이다. 실패하면 바로 예외가 나고 API 가 503 으로 응답한다.
# 이 설정은 agent.py 의 채팅 모델에서도 함께 쓴다.
BEDROCK_CONFIG = Config(read_timeout=30, connect_timeout=30, retries={"total_max_attempts": 1, "mode": "standard"})

# 만들어 둔 검색 인덱스 (모듈 전역 변수). None 이면 아직 만들지 않은 상태다.
# `FAISS | None` 은 "FAISS 객체 또는 null" 이라는 타입 힌트다. Python 은 타입 힌트를 실행 시 강제하지 않고, 읽는 사람과 도구를 위한 표시다.
_index: FAISS | None = None


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """마크다운 맨 위의 front matter(--- 로 감싼 메타데이터)에서 doc_id, title 을 읽고, 나머지 본문을 돌려준다.

    입력 예:
        ---
        doc_id: REG-01
        title: 방문 신청 절차
        ---
        # 방문 신청 절차 ...

    Returns:
        (메타데이터 dict, 본문 문자열) 튜플. Python 함수는 값 여러 개를 튜플로 한 번에 돌려줄 수 있다.
    """
    # split("---", 2): "---" 기준으로 최대 2번 잘라 3조각 [앞(빈 문자열), 헤더, 본문] 을 만든다.
    # 왼쪽에 변수를 여러 개 쓰면 조각을 순서대로 나눠 담는다(언패킹). _ 는 "안 쓰는 값" 이라는 관례 이름이다.
    _, header, body = raw.split("---", 2)
    meta = {}  # Java 의 new HashMap<>()
    for line in header.strip().splitlines():  # strip(): 앞뒤 공백 제거, splitlines(): 줄 단위 분리
        # partition(":") 은 첫 ':' 기준으로 (앞, ':', 뒤) 3개를 돌려준다. 가운데 구분자는 버린다.
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body.strip()


def load_documents(directory: Path = REGULATIONS_DIR) -> list[Document]:
    """규정 문서를 '## ' 섹션 단위 청크로 나눈다. 각 청크는 doc_id 와 title 을 메타데이터로 가진다.

    청크 본문 앞에는 "[문서 제목]" 을 붙인다. 섹션만 떼어 놓으면 어느 규정인지 알기 어렵기 때문이다.
    (화면의 근거 칩도 이 "[제목]" 을 읽어 문서 이름을 표시한다)

    Args:
        directory: 규정 문서 폴더. 기본값은 data/regulations (Java 와 달리 Python 은 인자 기본값을 직접 지정할 수 있다)
    """
    chunks = []
    # glob("*.md"): 폴더의 .md 파일 목록. sorted 로 파일명 순서를 고정해 결과가 매번 같게 한다.
    for path in sorted(directory.glob("*.md")):
        meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
        sections = body.split("\n## ")  # 줄 맨 앞의 "## " (2단계 제목) 기준으로 자른다
        # sections[0] 은 첫 "## " 앞부분(문서 제목, 안내 문장), sections[1:] 은 각 섹션이다
        intro, rest = sections[0], sections[1:]
        # `rest or [intro]`: 섹션이 하나도 없으면(빈 리스트는 False 취급) 앞부분 전체를 한 조각으로 쓴다
        for section in rest or [intro]:
            # split 하면서 사라진 "## " 를 다시 붙여 원래 제목 형태를 살린다
            text = section if not rest else f"## {section}"
            chunks.append(
                Document(
                    page_content=f"[{meta['title']}]\n{text.strip()}",
                    metadata={"doc_id": meta["doc_id"], "title": meta["title"]},
                )
            )
    return chunks


def default_embeddings() -> Embeddings:
    """기본 임베딩 모델: Amazon Titan Text Embeddings v2 (Bedrock, 30초 타임아웃·재시도 없음)."""
    return BedrockEmbeddings(model_id=EMBEDDING_MODEL_ID, region_name=REGION, config=BEDROCK_CONFIG)


def build_index(embeddings: Embeddings | None = None) -> FAISS:
    """규정 청크로 메모리 FAISS 인덱스를 만든다.

    Args:
        embeddings: 사용할 임베딩. None 이면 Titan(Bedrock)을 쓴다. 테스트에서는 가짜 임베딩을 넘겨 Bedrock 을 부르지 않는다.
    """
    # `embeddings or default_embeddings()`: 앞 값이 None 이면 뒤 함수를 호출한 결과를 쓴다
    return FAISS.from_documents(load_documents(), embeddings or default_embeddings())


def set_index(index: FAISS | None) -> None:
    """사용할 인덱스를 직접 지정한다. 테스트에서 가짜 임베딩 인덱스를 넣거나, None 으로 초기화할 때 쓴다."""
    global _index  # 모듈 전역 변수에 대입하려면 global 선언이 필요하다
    _index = index


def search(query: str, k: int = 3) -> list[dict[str, str]]:
    """질의와 의미가 가까운 규정 청크 k 개를 {doc_id, text} 목록으로 돌려준다.

    인덱스가 아직 없으면 이때 처음 만든다(지연 초기화). 만들 때 Bedrock 임베딩을 청크 수만큼 호출한다.

    Args:
        query: 검색 질문
        k: 돌려줄 조각 수 (기본 3)
    """
    global _index
    if _index is None:
        _index = build_index()
    # similarity_search: 질의 벡터와 가장 가까운 문서 k 개를 Document 목록으로 돌려준다.
    # 리스트 컴프리헨션으로 Document → {"doc_id": ..., "text": ...} dict 로 바꾼다 (Java 의 stream().map(...).toList()).
    return [{"doc_id": d.metadata["doc_id"], "text": d.page_content} for d in _index.similarity_search(query, k=k)]
