#!/bin/zsh
# 雙擊這個檔就能開 EDF Viewer。
# 已經在跑的話直接開瀏覽器，不會重複啟動。

APP="/Users/appletina/edf_viewer/app.py"
STREAMLIT="/Users/appletina/anaconda3/bin/streamlit"
PORT=8502

cd "$(dirname "$APP")" || exit 1

if curl -s -o /dev/null -m 2 "http://localhost:$PORT"; then
  echo "EDF Viewer 已經在跑，開瀏覽器…"
  open "http://localhost:$PORT"
  echo
  echo "（這個視窗可以直接關掉）"
  exit 0
fi

echo "啟動 EDF Viewer…"
echo "網址：http://localhost:$PORT"
echo
echo "★ 要關掉 viewer：在這個視窗按 Control + C，或直接關掉這個視窗。"
echo

"$STREAMLIT" run "$APP" --server.port "$PORT"
