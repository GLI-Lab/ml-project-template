# GitHub 템플릿 레포 만들기

이 문서는 `ml-project-template` 자체를 처음 만들고 GitHub 템플릿으로 등록하는 과정을 설명합니다.  
이미 템플릿을 사용해 새 프로젝트를 시작하는 경우라면 [README](../README.md)를 참고하세요.

## 사전 요구사항

- [Git](https://git-scm.com/): `sudo apt install git` (Ubuntu) / `brew install git` (macOS)
- [Pixi](https://pixi.sh/): `curl -fsSL https://pixi.sh/install.sh | sh`
- [Docker Engine](https://docs.docker.com/engine/install/): GPU 사용 시 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 추가 설치 (macOS는 [Docker Desktop](https://www.docker.com/products/docker-desktop/) 필요)

## 1. Git 사용자 설정

```bash
# 전역 설정 (모든 레포에 적용)
git config --global user.name "gli-lab"
git config --global user.email "glilab509@gmail.com"

# 또는 이 프로젝트에만 적용하려면 --global 빼고
git config user.name "gli-lab"
git config user.email "glilab509@gmail.com"

# 설정 확인
git config --list
```

## 2. 레포 생성 및 푸시

```bash
mkdir ml-project-template && cd ml-project-template
git init

# 파일 구성 후
git add .
git commit -m "initial template"

# GitHub 웹에서 이미 만든 레포와 연결
git remote add origin https://github.com/gli-lab/ml-project-template.git
git branch -M main
git push -u origin main
```

## 3. GitHub에서 템플릿으로 지정

1. GitHub 레포 페이지 → **Settings**
2. General 섹션 → **"Template repository"** 체크박스 ON

이후 레포 메인 페이지에 **"Use this template"** 버튼이 활성화됨.
