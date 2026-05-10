# 文档索引

`docs/` 目录只放项目说明和设计资料，避免在根目录堆放长文档。

## 推荐阅读顺序

1. `../README.md`：项目入口、快速运行、环境变量入口。
2. `current_capabilities.md`：当前能实现的功能、分步命令、已验证链路和待补全功能。
3. `report_modes.md`：整体转译、筛选说话人、筛选议题三种报告模式。
4. `diarization_backends.md`：pyannote 与 3D-Speaker 后端的使用方式。
5. `diarization_cleaning.md`：说话人分离结果为什么会碎片化，以及如何清洗。
6. `methodology_review.md`：原始方法论方案与当前实现之间的差异。
7. `project_structure.md`：上传 GitHub 前的项目树和提交规则。

## Archive

`archive/original_methodology.md` 是原始 POC 方案草稿，只作为设计来源保留。当前可运行实现以 `src/video_kb/` 和 `README.md` 为准。
