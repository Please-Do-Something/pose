# PC(데스크톱) 개발 환경 설정

노트북에서 세팅한 프로젝트를 다른 컴퓨터에서 이어서 작업하기 위한 가이드입니다.

## 1. 필수 프로그램 설치

- [Git](https://git-scm.com/downloads)
- [Python 3.10+](https://www.python.org/downloads/) (설치 시 "Add to PATH" 체크)
- [GitHub CLI](https://cli.github.com/) (`gh`)

## 2. GitHub 로그인

```
gh auth login
```

`GitHub.com` → `HTTPS` → `Login with a web browser` 순서로 선택 후, 표시되는 코드를 https://github.com/login/device 에서 입력.

노트북에서 사용한 것과 **같은 GitHub 계정(kimunet01)** 으로 로그인해야 합니다.

## 3. 저장소 클론

원하는 작업 폴더로 이동한 뒤:

```
gh repo clone Please-Do-Something/pose
cd pose
```

## 4. 가상환경 생성 및 패키지 설치

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 5. 실행

```
python main.py
```

## 6. 앞으로의 작업 흐름 (노트북 ↔ PC 동기화)

**작업 시작 전 (항상 먼저):**
```
git pull
```

**작업 끝난 후:**
```
git add -A
git commit -m "작업 내용"
git push
```

두 기기 모두 작업 시작 전 `pull`, 끝나고 `push` 하는 습관만 지키면 항상 최신 상태로 동기화됩니다.

## 7. Claude Code CLI 설치 (선택)

노트북과 동일하게 Claude Code를 PC에서도 쓰려면:

```
npm install -g @anthropic-ai/claude-code
```

(Node.js가 없다면 먼저 https://nodejs.org 에서 설치)

설치 후 프로젝트 폴더에서 실행:

```
cd pose
claude
```

처음 실행 시 브라우저로 로그인 화면이 뜨며, 노트북에서 쓰던 것과 **같은 Anthropic 계정**으로 로그인하면 됩니다. 계정 로그인은 기기별로 따로 하는 것이라 대화 기록 자체가 동기화되진 않지만, 프로젝트 코드는 git으로 동기화되어 있으니 그대로 이어서 작업 지시하면 됩니다.

## 참고: 동기화되지 않는 파일

아래 파일/폴더는 `.gitignore`에 의해 제외되어 있어 기기마다 로컬로 따로 존재합니다:

- `venv/` — 가상환경 (기기마다 새로 생성)
- `__pycache__/` — 파이썬 캐시
- `posture.db` — 자세 기록 데이터베이스 (기기별 로컬 데이터)

`posture.db`까지 두 기기에서 공유하고 싶다면 별도 동기화 방법(예: 클라우드 스토리지, DB 서버 전환 등)이 필요합니다.
