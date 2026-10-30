# Git发布

本次只整理并初始化本地源码仓库；未提交、设置远端或推送，原数据没有删除。

Windows若提示dubious ownership，请先核对项目确属自己的工作区。可在每次git命令前添加`-c safe.directory=项目绝对路径`只信任当前调用，例如`git -c safe.directory=D:/AILearning/project_for_offer/FitnessPoseCoach status`。本次未修改全局safe.directory，也不要设置为通配符。检查脚本只对自身项目目录设置单次信任。

提交源码、配置、测试、依赖、公开文档及third_party；排除环境、数据、模型、outputs及releases实物、本机会话AGENTS.md及历史实验记录。旧docs原位保留但忽略，公开文档采用.gitignore显式清单；新增公开文档要更新清单。

.gitattributes关闭换行转换，以免改变冻结字节指纹。不批量格式化冻结源码/JSON。项目不是pip发行包，不虚构`pip install fitnessposecoach`。

## 检查

```bash
python tools/check_repository.py
git status --short
git add --dry-run .
```

检查只读Git候选，拒绝超过1MiB的大文件、常见敏感路径/私钥/令牌模式，不是全面安全审计。历史中已有秘密不能只靠ignore清除；本次为空历史。

## 自己创建远端后

```bash
git add .
git diff --cached --stat
git commit -m "Initial source release"
# 换成自己的仓库地址，不要原样使用占位地址。
git remote add origin https://github.com/YOUR_ACCOUNT/FitnessPoseCoach.git
git push -u origin main
```

需要身份时配置自己的git姓名/邮箱，不使用他人身份。本次未代用户选择顶层开源许可；见[许可说明](LICENSING.md)。

GitHub Actions只运行合成CPU检查，不下载数据、权重或访问相机。克隆后可跑CPU测试，真实相机使用仍需[恢复产物](DATA_AND_MODELS.md)。
