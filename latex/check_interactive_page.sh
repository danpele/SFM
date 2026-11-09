#!/bin/zsh
# usage: cd 2026/interactive && ../latex/check_interactive_page.sh chN_ro.html            (interactive page)
#        cd 2026/web         && ../latex/check_interactive_page.sh chN_ro.html web [390]  (web lecture; optional window width)
# needs a server on localhost:8765 serving the SFM repo root 2026/ (python3 -m http.server 8765).
# Prints JS errors, empty readouts, number of plots, MathJax state and page width (window/document; equal = no horizontal overflow).
SP=${TMPDIR:-/tmp}
f=$1; dir=${2:-interactive}; w=${3:-1440}; t=_check_$RANDOM.html
python3 -c "
s=open('$f',encoding='utf-8').read().replace('</body>','<script>window.__err=[];addEventListener(\"error\",e=>window.__err.push(e.message));setTimeout(()=>{document.body.setAttribute(\"data-err\",JSON.stringify(window.__err)+\" empty=\"+[...document.querySelectorAll(\".out\")].filter(u=>!u.innerHTML).map(u=>u.id).join(\",\")+\" plots=\"+document.querySelectorAll(\".main-svg\").length+\" charts=\"+document.querySelectorAll(\".chart\").length+\" mj=\"+(document.body.getAttribute(\"data-mj\")||\"-\")+\" mjerr=\"+document.querySelectorAll(\"mjx-merror\").length+\" w=\"+innerWidth+\"/\"+document.documentElement.scrollWidth)},15000)</script></body>')
s=s.replace('<head>','<head><script>window.__err=[];addEventListener(\"error\",e=>(window.__err=window.__err||[]).push(e.message));</script>',1)
open('$t','w',encoding='utf-8').write(s)"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --user-data-dir=$SP/chrome-prof-$RANDOM --window-size=$w,900 --virtual-time-budget=25000 --dump-dom "http://localhost:8765/$dir/$t" 2>/dev/null | grep -o 'data-err="[^"]*"'
python3 -c "import os; os.remove('$t')"
