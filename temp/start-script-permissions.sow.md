# SOW: start.sh / compose 権限整備

**作成日**: 2025-11-03  
**プロジェクト**: MiniRAG  
**作業対象**: `scripts/start.sh` の権限ハンドリングと Docker Compose ボリューム設定

---

## 1. 背景と目的

- `sudo scripts/start.sh cleanup` 実行時に `CURRENT_UID=$(id -u)` が `0` を返し、`./postgres` / `./postgres/sql` 配下が `root:root` 所有となる事象が発生。
- `docker compose` のボリューム定義が読み書き可能マウントのため、コンテナ側プロセスがホスト上の SQL ファイルに対して所有権変更を行う余地がある。
- 目的は、**ホスト開発ユーザーが常に書き込み可能**な状態を維持しつつ、PostgreSQL コンテナからのアクセス要件（UID 999）も満たせるよう、スクリプトと Compose 設定を整備すること。

---

## 2. スコープ

### 2.1 対象作業

1. **start.sh の所有者判定ロジック修正**  
   - `SUDO_UID` / `SUDO_GID` を優先利用し、`sudo` 実行時でもホストユーザーを正しく認識するようにする。  
   - `chown` / `chmod` 対象ディレクトリを明示し、`./postgres/sql` 配下が `root` 化しないよう調整。  
   - ログメッセージで実際に設定する UID/GID を可視化し、トラブルシュート性を向上させる。

2. **Docker Compose のボリュームマウント見直し**  
   - `./postgres/init` および `./postgres/sql/migrations` のマウントを読み取り専用 (`:ro`) に変更し、コンテナ側からの不意の所有権変更を抑止。  
   - （必要に応じて）`postgres` サービスの実行ユーザー設定を確認し、UID 999 前提の権限整備と齟齬がないか検証。

3. **動作確認**  
   - `scripts/start.sh cleanup` → `scripts/stop.sh` を `sudo` なし/あり両パターンで試行し、`./postgres/sql` と `./data/postgres` の所有権が期待通りか確認。  
   - `docker compose` 起動後の初期化処理が失敗しないことをログで確認。

### 2.2 非スコープ

- PostgreSQL マイグレーション内容の更新。
- 既存ドキュメント (`README.md` 等) の運用手順全面改訂。必要最小限の補足に留める。
- `stop.sh` / `build.sh` の新規機能追加。

---

## 3. 成果物

| 種別 | パス | 内容 |
|---|---|---|
| スクリプト | `scripts/start.sh` | UID/GID の判定ロジックと権限調整処理の更新 |
| 設定 | `compose.yaml` | 初期化用マウントの読み取り専用化、必要に応じたユーザー設定見直し |
| 検証記録 | `temp/` 以下メモ or 実行ログ | 所有権確認結果とコンテナ起動ログ抜粋 |

---

## 4. タスクと見積

| ID | タスク | 詳細 | 見積時間 |
|---|---|---|---|
| T1 | start.sh 修正 | UID/GID 判定、`chown` 対象整理、ログ整備 | 1.0 h |
| T2 | compose.yaml 更新 | ボリュームの `:ro` 化、ユーザー設定確認 | 0.5 h |
| T3 | 動作検証 | `cleanup` 有無＋`sudo` 有無で所有権/動作確認 | 0.5 h |

---

## 5. リスクと緩和策

| リスク | 影響 | 緩和策 |
|---|---|---|
| `SUDO_UID` が未定義な環境 | 所有権が再び `root` 化 | `id -u` / `id -g` をフォールバックとして実装 |
| 読み取り専用マウントにより初期化処理が失敗 | DB 初期化エラー | 事前に手動＆自動テストを実施し、必要なら README に手順追記 |
| Docker 実行ユーザーとの不一致 | コンテナ起動失敗 | `docker compose` 起動ログで UID/GID エラーを確認し、必要なら設定調整 |

---

## 6. 成功基準

- `sudo scripts/start.sh cleanup` 実行後でも `./postgres/sql` と `./postgres/init` がホストユーザー所有で維持される。  
- `./data/postgres` はホストユーザー:UID / グループ:999 で作成され、PostgreSQL が問題なく書き込める。  
- 変更後の `docker compose up -d` が正常完了し、初期化スクリプトが従来通り実行される。  
- README 既存手順（必要な場合の軽微な追記含む）に矛盾が生じない。

---

## 7. 承認ポイント

1. スコープ（T1〜T3）が期待に合致しているか。  
2. start.sh での UID/GID 判定方針が妥当か。  
3. Compose ボリュームを読み取り専用にする方針に問題がないか。  
4. 動作検証レベル（cleanup 有無・sudo 有無）が十分か。


