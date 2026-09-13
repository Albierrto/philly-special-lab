"""Compose the v4 app-shell template from the v3 sections + JS. python -m breakout.build_v4"""
from __future__ import annotations
import re, sys
from . import config as C

SHELL_HEAD = r'''<title>Philly Special Hitter Lab</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{
  --bg:#0D1726; --panel:#142138; --panel2:#1B2B47; --panel3:#243755; --line:#2A3C58; --ink:#EEF3F9; --ink2:#B5C2D4; --ink3:#7E8FA7;
  --accent:#E5484D; --amber:#F0B429; --hot:#F0645C; --cold:#5FA8F5; --good:#4CC27A; --bad:#F26D6D; --warn:#E8B04B; --mine:#3A3110; --mine-ink:#F7E3A1;
  --hot-rgb:240,100,92; --cold-rgb:95,168,245; --shadow:0 10px 30px rgba(0,0,0,.35);
  --font-d:"Barlow Condensed","Arial Narrow",Impact,sans-serif; --font-b:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif; --font-m:"IBM Plex Mono",Menlo,Consolas,monospace;
  color-scheme:dark;
}
@media (prefers-color-scheme: light){ :root:not([data-theme="dark"]){
  --bg:#EEF1F5; --panel:#FFFFFF; --panel2:#E8ECF2; --panel3:#DAE0E9; --line:#D3DAE4; --ink:#121B2B; --ink2:#46546A; --ink3:#77849A;
  --accent:#C8232A; --amber:#B7791F; --hot:#C8312B; --cold:#2C6FB7; --good:#2E7D4F; --bad:#B23A2E; --warn:#B7791F; --mine:#FBEFC4; --mine-ink:#5A4300;
  --hot-rgb:200,49,43; --cold-rgb:44,111,183; --shadow:0 10px 30px rgba(20,30,50,.12); color-scheme:light; }}
:root[data-theme="light"]{
  --bg:#EEF1F5; --panel:#FFFFFF; --panel2:#E8ECF2; --panel3:#DAE0E9; --line:#D3DAE4; --ink:#121B2B; --ink2:#46546A; --ink3:#77849A;
  --accent:#C8232A; --amber:#B7791F; --hot:#C8312B; --cold:#2C6FB7; --good:#2E7D4F; --bad:#B23A2E; --warn:#B7791F; --mine:#FBEFC4; --mine-ink:#5A4300;
  --hot-rgb:200,49,43; --cold-rgb:44,111,183; --shadow:0 10px 30px rgba(20,30,50,.12); color-scheme:light; }
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font-b);font-size:13.5px;line-height:1.45;padding-inline:0;padding-block:0}
a{color:var(--cold)}
button{font-family:inherit}
/* ---------- app shell */
.topbar{position:sticky;top:0;z-index:30;height:56px;display:flex;align-items:center;gap:14px;padding:0 16px;background:var(--panel);border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:10px;text-decoration:none;color:var(--ink);white-space:nowrap}
.brand .mark{width:30px;height:30px;flex:0 0 auto}
.brand b{font-family:var(--font-d);font-weight:800;font-size:22px;letter-spacing:.06em;text-transform:uppercase;line-height:1}
.brand b span{color:var(--accent)}
.brand small{display:block;font-family:var(--font-m);font-size:10px;letter-spacing:.14em;color:var(--ink3);text-transform:uppercase;margin-top:2px}
.teamsel{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--ink3);white-space:nowrap}
.teamsel select{font-size:12.5px;padding:5px 8px;border-radius:999px;max-width:190px}
.search{position:relative;flex:1;max-width:460px;margin-left:auto}
.search input[type="search"]{width:100%;font:inherit;font-size:13px;padding:8px 12px 8px 34px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink)}
.search svg{position:absolute;left:11px;top:9px;width:16px;height:16px;color:var(--ink3)}
.search .dd{position:absolute;top:40px;left:0;right:0;background:var(--panel);border:1px solid var(--line);border-radius:8px;box-shadow:var(--shadow);display:none;overflow:hidden;z-index:40}
.search .dd.open{display:block}
.search .dd button{display:flex;width:100%;align-items:center;gap:10px;background:none;border:0;text-align:left;padding:8px 12px;color:var(--ink);cursor:pointer;font-size:13px}
.search .dd button:hover,.search .dd button:focus{background:var(--panel2);outline:none}
.search .dd .ava{width:26px;height:26px;font-size:10px}
.search .dd small{color:var(--ink3);margin-left:auto;font-family:var(--font-m);font-size:11px}
.liveind{font-family:var(--font-m);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3);white-space:nowrap;cursor:pointer}
.liveind.ok{color:var(--good)} .liveind.busy{color:var(--amber)} .liveind.err{color:var(--bad)}
.liveind::before{content:"●";margin-right:5px}
.liveind:empty{display:none}
.iconbtn{width:36px;height:36px;border:1px solid var(--line);background:var(--panel);color:var(--ink2);border-radius:8px;display:inline-flex;align-items:center;justify-content:center;cursor:pointer}
.iconbtn svg{width:18px;height:18px}
.iconbtn:hover{color:var(--ink);background:var(--panel2)}
.iconbtn[aria-pressed="true"]{color:var(--accent);border-color:var(--accent)}
.shell{display:grid;grid-template-columns:212px 1fr;min-height:calc(100vh - 56px)}
.rail{position:sticky;top:56px;height:calc(100vh - 56px);overflow:auto;border-right:1px solid var(--line);background:var(--panel);padding:12px 10px;display:flex;flex-direction:column;gap:2px}
.rail .grp{font-family:var(--font-m);font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3);padding:12px 10px 4px}
.navbtn{display:flex;align-items:center;gap:10px;width:100%;border:0;background:transparent;color:var(--ink2);padding:9px 10px;border-radius:8px;cursor:pointer;font-size:13.5px;font-weight:500;text-align:left}
.navbtn svg{width:18px;height:18px;flex:0 0 auto;opacity:.9}
.navbtn:hover{background:var(--panel2);color:var(--ink)}
.navbtn[aria-selected="true"]{background:var(--panel2);color:var(--ink);box-shadow:inset 3px 0 0 var(--accent)}
.rail .foot{margin-top:auto;padding:10px;font-size:11px;color:var(--ink3);line-height:1.4}
main{min-width:0;padding:18px clamp(14px,2.2vw,28px) 90px}
.tabbar{display:none}
section[role="tabpanel"]{display:none} section[role="tabpanel"].on{display:block}
.pagehead{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:8px 16px;margin:0 0 14px}
h1,h2{font-family:var(--font-d);font-weight:700;text-transform:uppercase;letter-spacing:.02em;line-height:1;margin:0}
h1{font-size:34px} h2{font-size:28px;margin:0 0 6px}
h3{font-family:var(--font-d);font-size:18px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;margin:16px 0 6px;color:var(--ink)}
.lede{color:var(--ink2);max-width:80ch;margin:0 0 12px;font-size:13.5px}
.lede b,.note b{color:var(--ink)}
/* ---------- settings drawer */
.drawer{position:fixed;top:56px;right:0;bottom:0;width:min(420px,100%);background:var(--panel);border-left:1px solid var(--line);box-shadow:var(--shadow);z-index:35;transform:translateX(100%);transition:transform .2s;padding:16px;overflow:auto}
.drawer.open{transform:none}
@media (prefers-reduced-motion:reduce){.drawer,.card{transition:none}}
.drawer h2{font-size:22px;margin-bottom:10px}
.leaguebar{display:flex;flex-direction:column;gap:10px}
.leaguebar label{font-size:12.5px;color:var(--ink2);display:flex;align-items:center;justify-content:space-between;gap:8px}
select,input[type="number"],input[type="search"],input[type="text"]{font:inherit;font-size:13px;padding:6px 8px;border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:6px}
input[type="number"]{width:68px;font-family:var(--font-m);font-size:12px}
.wts{display:flex;flex-wrap:wrap;gap:6px 10px}
.wts label{font-family:var(--font-m);font-size:11.5px;color:var(--ink2);display:flex;align-items:center;gap:4px}
.wts input{width:54px;padding:3px 4px}
.btn{font:inherit;font-size:12.5px;font-weight:500;border:1px solid var(--line);background:var(--panel);color:var(--ink);padding:6px 11px;cursor:pointer;border-radius:6px}
.btn:hover{background:var(--panel2)}
.btn.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.btn.small{padding:4px 9px;font-size:12px}
.seg{display:inline-flex;border:1px solid var(--line);background:var(--bg);border-radius:6px;overflow:hidden}
.seg button{font:inherit;font-size:12.5px;font-weight:500;border:0;background:transparent;color:var(--ink2);padding:6px 11px;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--ink);color:var(--bg)}
/* ---------- toolbars and filters */
.filters{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center;margin-bottom:10px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 12px}
.filters label{font-size:12px;color:var(--ink2);display:flex;align-items:center;gap:6px}
.chips{display:flex;flex-wrap:wrap;gap:4px}
.chip{font-size:12px;border:1px solid var(--line);background:var(--bg);color:var(--ink2);padding:2px 9px;border-radius:999px;cursor:pointer;user-select:none}
.chip[aria-pressed="true"]{background:var(--cold);color:#fff;border-color:var(--cold)}
.rangefilters{display:flex;flex-direction:column;gap:6px;margin:6px 0 10px}
.rf{display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:12px;color:var(--ink2)}
.rf select{min-width:180px}
.colpicker{display:none;flex-wrap:wrap;gap:4px 16px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:10px}
.colpicker.open{display:flex}
.colpicker .grp{min-width:150px}
.colpicker .grp b{display:block;font-family:var(--font-d);font-size:14px;text-transform:uppercase;letter-spacing:.04em;margin-bottom:2px}
.colpicker label{display:block;font-size:12px;color:var(--ink2);cursor:pointer}
.hint{color:var(--ink3);font-size:12px}
.count{font-family:var(--font-m);font-size:12px;color:var(--ink3)}
/* ---------- tables */
.tblwrap{overflow:auto;border:1px solid var(--line);border-radius:10px;background:var(--panel);max-height:72vh}
table{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%;font-variant-numeric:tabular-nums;font-size:12.5px}
th{position:sticky;top:0;z-index:2;background:var(--panel2);color:var(--ink2);font-weight:600;text-align:right;padding:8px 8px;border-bottom:1px solid var(--line);white-space:nowrap;cursor:pointer;user-select:none;font-size:11px;letter-spacing:.04em;text-transform:uppercase}
th.l,td.l{text-align:left}
th.sticky,td.sticky{position:sticky;left:0;z-index:3;background:var(--panel);box-shadow:1px 0 0 var(--line)}
th.sticky{z-index:4;background:var(--panel2)}
th[aria-sort="descending"]::after{content:" ▾";color:var(--accent)} th[aria-sort="ascending"]::after{content:" ▴";color:var(--accent)}
td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap;font-family:var(--font-m);font-size:12px}
td.l{font-family:var(--font-b);font-size:12.5px}
tbody tr:hover td{background-color:var(--panel2)}
tbody tr:hover td.sticky{background:var(--panel2)}
tr.mine td.sticky{background:var(--mine)} tr.mine td.sticky .sub{color:var(--mine-ink)}
td.name{font-weight:600;cursor:pointer}
td.name .sub{display:block;font-weight:400;color:var(--ink3);font-size:11px;font-family:var(--font-b)}
.pill{display:inline-block;padding:0 6px;border-radius:4px;font-size:11px;font-weight:600;font-family:var(--font-b);line-height:18px}
.pill.g{background:rgba(76,194,122,.18);color:var(--good)} .pill.r{background:rgba(var(--hot-rgb),.18);color:var(--hot)} .pill.b{background:rgba(var(--cold-rgb),.18);color:var(--cold)} .pill.w{background:rgba(232,176,75,.18);color:var(--warn)} .pill.n{background:var(--panel3);color:var(--ink2)}
.pos-d{color:var(--good);font-weight:600}.neg-d{color:var(--bad);font-weight:600}
/* ---------- dashboard */
.dash{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}
.dcard{grid-column:span 4;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;min-width:0}
.dcard.w6{grid-column:span 6} .dcard.w8{grid-column:span 8} .dcard.w12{grid-column:span 12}
.dcard h3{margin:0 0 8px;display:flex;align-items:baseline;justify-content:space-between;gap:8px}
.dcard h3 small{font-family:var(--font-b);font-size:11px;color:var(--ink3);text-transform:none;letter-spacing:0;font-weight:400}
.dcard .go{font-size:12px;color:var(--cold);background:none;border:0;cursor:pointer;padding:0;font-family:var(--font-b)}
.rowlist{display:flex;flex-direction:column}
.row{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:10px;padding:6px 0;border-top:1px solid var(--line);cursor:pointer}
.row:first-child{border-top:0}
.row:hover .nm{color:var(--cold)}
.row>div{min-width:0}
.row .nm{font-weight:600;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.row .sub{font-size:11.5px;color:var(--ink3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.row .val{font-family:var(--font-m);font-size:13px;font-weight:600;text-align:right}
.row .val small{display:block;font-weight:400;color:var(--ink3);font-size:10.5px}
.ava{width:32px;height:32px;border-radius:50%;background:var(--tc,var(--panel3));color:#fff;display:inline-flex;align-items:center;justify-content:center;font-family:var(--font-d);font-weight:700;font-size:12px;letter-spacing:.02em;flex:0 0 auto;box-shadow:inset 0 0 0 2px rgba(255,255,255,.12)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(112px,1fr));gap:8px;margin:10px 0}
.kpi{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.kpi b{display:block;font-family:var(--font-d);font-size:24px;font-weight:700;line-height:1}
.kpi span{font-size:10.5px;color:var(--ink3);text-transform:uppercase;letter-spacing:.06em}
.stat{display:flex;align-items:baseline;gap:8px}
.stat b{font-family:var(--font-d);font-size:34px;font-weight:700;line-height:1}
.stat span{color:var(--ink3);font-size:12px}
.todaystrip{display:flex;flex-wrap:wrap;gap:8px}
.wxchip{display:flex;flex-direction:column;gap:2px;background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:6px 10px;min-width:118px;font-size:12px}
.wxchip b{font-family:var(--font-d);font-size:15px;font-weight:600;letter-spacing:.02em}
.wxchip small{color:var(--ink3);font-size:11px}
/* ---------- player page (slide-over) */
.overlay{position:fixed;inset:0;background:rgba(3,8,16,.55);display:none;z-index:40}
.overlay.open{display:block}
.card{position:fixed;top:0;right:0;bottom:0;width:min(780px,100%);background:var(--bg);z-index:41;overflow:auto;transform:translateX(100%);transition:transform .2s;padding:0 0 30px;border-left:1px solid var(--line)}
.card.open{transform:none}
.card .close{position:absolute;top:12px;right:14px;z-index:2}
.card-body{padding:0 clamp(14px,2vw,24px) 20px}
.pband{display:flex;align-items:center;gap:14px;padding:22px clamp(14px,2vw,24px) 16px;background:linear-gradient(135deg,var(--tc,#33415c) 0%,color-mix(in srgb,var(--tc,#33415c) 45%,var(--bg)) 70%,var(--bg) 100%);border-bottom:1px solid var(--line)}
.pband .ava{width:56px;height:56px;font-size:20px;box-shadow:inset 0 0 0 3px rgba(255,255,255,.25)}
.pband h2{font-size:32px;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.35)}
.pband .hint{color:rgba(255,255,255,.85);font-size:12.5px}
.pband .pill{background:rgba(255,255,255,.18);color:#fff}
.card h3{margin-top:18px}
.card table{font-size:12px}
.bars{display:flex;gap:2px;align-items:flex-end;height:36px}
.bars i{display:block;width:14px;background:var(--cold);opacity:.85;border-radius:2px 2px 0 0}
.bars i.cur{background:var(--hot)}
.zrow{display:grid;grid-template-columns:150px 1fr 60px;gap:8px;align-items:center;font-size:12px;margin:2px 0}
.pbar{position:relative;height:16px;margin:0 16px 0 12px;background:linear-gradient(90deg,rgba(var(--cold-rgb),.35),var(--panel2) 50%,rgba(var(--hot-rgb),.35));border-radius:8px}
.pbar b{position:absolute;top:-3px;width:22px;height:22px;margin-left:-11px;border-radius:50%;color:#fff;font-size:10px;font-weight:600;display:flex;align-items:center;justify-content:center;font-family:var(--font-m);box-shadow:0 0 0 2px var(--bg)}
.pbar i{position:absolute;top:2px;bottom:2px;width:2px}
.pas{display:flex;height:18px;border-radius:4px;overflow:hidden;background:var(--panel2);margin:6px 0 2px}
.pas span{display:flex;align-items:center;justify-content:center;font-size:10.5px;color:#fff;font-family:var(--font-m);white-space:nowrap;overflow:hidden}
.pas .p1{background:#2C6FB7}.pas .p2{background:#2E7D4F}.pas .p3{background:#B7791F}.pas .p4{background:#C8312B}
details{border:1px solid var(--line);background:var(--panel);padding:6px 10px;margin:6px 0;border-radius:8px} summary{cursor:pointer;font-size:13px} table.mini{width:100%;margin-top:6px} table.mini td{padding:3px 6px;font-size:12px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
.note{border-left:3px solid var(--accent);padding:8px 12px;background:var(--panel);margin:10px 0;max-width:80ch;color:var(--ink2)}
.formula{font-family:var(--font-m);font-size:12px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:6px 0 10px;white-space:pre-wrap;color:var(--ink)}
.method p,.method li{max-width:76ch;color:var(--ink2)} .method li b,.method p b{color:var(--ink)}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--ink2);margin:6px 0 0}
.legend span::before{content:"";display:inline-block;width:10px;height:10px;margin-right:5px;vertical-align:-1px;border-radius:2px}
.legend .h::before{background:var(--hot)} .legend .c::before{background:var(--cold)} .legend .m::before{background:var(--mine);border:1px solid var(--line)}
.legend span:not(.h):not(.c):not(.m)::before{display:none}
svg text{fill:var(--ink2);font-family:var(--font-m);font-size:11px}
button:focus-visible,select:focus-visible,input:focus-visible,.chip:focus-visible{outline:2px solid var(--cold);outline-offset:1px}
/* ---------- responsive */

@media (max-width:900px){
  .shell{grid-template-columns:1fr} .rail{display:none}
  .tabbar{display:flex;position:fixed;left:0;right:0;bottom:0;z-index:30;background:var(--panel);border-top:1px solid var(--line);overflow-x:auto;padding:4px 4px calc(4px + env(safe-area-inset-bottom))}
  .tabbar .navbtn{flex:1 0 64px;flex-direction:column;gap:3px;font-size:10.5px;padding:6px 4px;border-radius:8px;text-align:center;justify-content:center}
  .tabbar .navbtn[aria-selected="true"]{box-shadow:none;color:var(--accent)}
  .dcard,.dcard.w6,.dcard.w8{grid-column:span 12}
  .zrow{grid-template-columns:110px 1fr 50px} .tblwrap{max-height:none}
  .teamsel span{display:none} .teamsel select{max-width:120px}
  .search{max-width:none;min-width:0} .search input[type="search"]{padding-left:30px} .brand small{display:none} .brand b{font-size:18px} .topbar{gap:8px;padding:0 10px} h1{font-size:28px}
  .row{grid-template-columns:auto minmax(0,1fr) auto} .row .sub{white-space:normal}
}
</style>
'''

ICONS = {
 'dash': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 11l9-8 9 8"/><path d="M5 10v10h14V10"/></svg>',
 'explore': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20l9-9"/><path d="M12 12l5-5a2.5 2.5 0 0 1 3.5 3.5l-5 5"/><circle cx="6" cy="18" r="1.5"/></svg>',
 'breakouts': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l6-6 4 4 8-8"/><path d="M14 7h7v7"/></svg>',
 'pitchers': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M6 6c3 2 3 10 0 12M18 6c-3 2-3 10 0 12"/></svg>',
 'streamers': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/></svg>',
 'keepers': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></svg>',
 'drafts': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6h16M4 12h16M4 18h10"/></svg>',
 'prospects': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 21V11"/><path d="M12 11c0-4 3-7 7-7 0 4-3 7-7 7z"/><path d="M12 14c0-3-2.5-5-5-5 0 3 2.5 5 5 5z"/></svg>',
 'formulas': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M18 5H7l6 7-6 7h11"/></svg>',
}
NAV = [('dash','Dashboard'),('explore','Hitters'),('breakouts','Breakouts'),('pitchers','Pitchers'),('streamers','Streamers'),('keepers','Keepers'),('drafts','Drafts'),('prospects','Prospects'),('formulas','Methods')]

MARK = '<svg class="mark" viewBox="0 0 32 32" aria-hidden="true"><path d="M4 4h24v12L16 30 4 16z" fill="#E5484D"/><path d="M8 8h16v7l-8 10-8-10z" fill="#fff" opacity=".92"/><path d="M12 12h8v3l-4 5-4-5z" fill="#E5484D"/></svg>'


def shell(sections: str) -> str:
    rail = ''.join(f'<button class="navbtn" role="tab" aria-selected="{"true" if k=="dash" else "false"}" data-tab="{k}">{ICONS[k]}<span>{l}</span></button>' for k, l in NAV)
    dash = r'''<section role="tabpanel" id="tab-dash" class="on">
  <div class="pagehead"><div><h1 id="dash-title">Dashboard</h1><p class="lede" id="dash-sub"></p></div><div class="hint" id="dash-asof"></div></div>
  <div class="dash" id="dash"></div>
</section>
'''
    drawer = r'''<aside class="drawer" id="settings" aria-label="League settings">
  <button class="btn small" id="settings-close" style="position:absolute;top:12px;right:14px">Close ✕</button>
  <h2>League settings</h2>
  <p class="hint">Everything re-scores when you change these. Presets approximate each site's default points; edit the weights for your league.</p>
  <div class="leaguebar" id="leaguebar">
    <label>Scoring preset <select id="preset"></select></label>
    <label>Mode <span class="seg" role="group"><button aria-pressed="true" data-mode="points" id="mode-points">Points</button><button aria-pressed="false" data-mode="roto" id="mode-roto">5x5 Roto</button></span></label>
    <label>Teams <input type="number" id="teams" value="12" min="4" max="30"></label>
    <label>Hitters started <input type="number" id="hstart" value="10" min="5" max="16"></label>
    <label>SP started <input type="number" id="pstart" value="5" min="1" max="12"></label>
    <label>Hitter keepers per team <input type="number" id="hkeep" value="5" min="0" max="10"></label>
    <label>Pitcher keepers per team <input type="number" id="pkeep" value="2" min="0" max="10"></label>
    <span class="hint" id="score-note"></span>
    <button class="btn small" id="toggle-wts">Edit weights</button>
  </div>
  <div id="wts-wrap" hidden><div class="hint" style="margin:8px 0 4px">Hitting</div><div class="wts" id="wts"></div><div class="hint" style="margin:10px 0 4px">Pitching <select id="ppreset" style="font-size:12px;padding:2px 4px"></select></div><div class="wts" id="pwts"></div></div>
</aside>
'''
    return SHELL_HEAD + f'''
<header class="topbar">
  <a class="brand" href="#" id="brand">{MARK}<div><b>Philly <span>Special</span></b><small>Lab · 2026 season</small></div></a>
  <label class="teamsel"><span>Viewing as</span><select id="team-sel" aria-label="Your team"></select></label>
  <div class="search"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg><input type="search" id="gsearch" placeholder="Find a player" aria-label="Find a player" autocomplete="off"><div class="dd" id="gsearch-dd" role="listbox"></div></div>
  <span class="liveind" id="live-ind" title="Live data from MLB and Open-Meteo refreshes every 10 minutes on the website; the artifact copy is the daily build"></span>
  <button class="iconbtn" id="theme-btn" title="Day / night" aria-label="Toggle theme"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg></button>
  <button class="iconbtn" id="settings-btn" title="League settings" aria-label="League settings"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg></button>
</header>
<div class="shell">
<nav class="rail" role="tablist" aria-label="Sections">{rail}<div class="foot">Scored in your league's points. Data: Baseball Savant, MLB Stats API, Fantrax (read-only), Open-Meteo.</div></nav>
<main>
{dash}{sections}</main>
</div>
<nav class="tabbar" role="tablist" aria-label="Sections">{rail}</nav>
{drawer}
<div class="overlay" id="overlay"></div>
<aside class="card" id="card" aria-label="Player page"><button class="btn small close" id="card-close">Close ✕</button><div id="card-body" class="card-body"></div></aside>
'''


JS_EXTRA = r'''
// ---------- team colors, avatars, player page header
const TEAM_COLORS = {ARI:'#A71930',ATH:'#0F5C3F',OAK:'#0F5C3F',ATL:'#CE1141',BAL:'#DF4601',BOS:'#BD3039',CHC:'#0E3386',CWS:'#3B3B3B',CHW:'#3B3B3B',CIN:'#C6011F',CLE:'#00385D',COL:'#4B3F8C',DET:'#0C2340',HOU:'#EB6E1F',KC:'#004687',KCR:'#004687',LAA:'#BA0021',LAD:'#005A9C',MIA:'#00A3E0',MIL:'#12284B',MIN:'#002B5C',NYM:'#002D72',NYY:'#1C2B4A',PHI:'#E81828',PIT:'#2B2B2B',SD:'#5C3D2E',SDP:'#5C3D2E',SF:'#FD5A1E',SFG:'#FD5A1E',SEA:'#0C5C5C',STL:'#C41E3A',TB:'#092C5C',TBR:'#092C5C',TEX:'#003278',TOR:'#134A8E',WSH:'#AB0003',WSN:'#AB0003'};
const TEAM_NAME_ABBR = {'Arizona Diamondbacks':'ARI','Athletics':'ATH','Oakland Athletics':'ATH','Atlanta Braves':'ATL','Baltimore Orioles':'BAL','Boston Red Sox':'BOS','Chicago Cubs':'CHC','Chicago White Sox':'CWS','Cincinnati Reds':'CIN','Cleveland Guardians':'CLE','Colorado Rockies':'COL','Detroit Tigers':'DET','Houston Astros':'HOU','Kansas City Royals':'KC','Los Angeles Angels':'LAA','Los Angeles Dodgers':'LAD','Miami Marlins':'MIA','Milwaukee Brewers':'MIL','Minnesota Twins':'MIN','New York Mets':'NYM','New York Yankees':'NYY','Philadelphia Phillies':'PHI','Pittsburgh Pirates':'PIT','San Diego Padres':'SD','San Francisco Giants':'SF','Seattle Mariners':'SEA','St. Louis Cardinals':'STL','Tampa Bay Rays':'TB','Texas Rangers':'TEX','Toronto Blue Jays':'TOR','Washington Nationals':'WSH'};
const abbrOf = t => !t ? '' : (TEAM_COLORS[t] ? t : (TEAM_NAME_ABBR[t] || (t.length<=3 ? t : t.split(' ').pop().slice(0,3).toUpperCase())));
const teamColor = t => TEAM_COLORS[abbrOf(t)] || '#33415C';
const initials = n => (n||'').split(' ').filter(x=>x && !/^(jr|sr|ii|iii)\.?$/i.test(x)).map(x=>x[0]).slice(0,2).join('').toUpperCase();
const ava = (name, team, size) => `<span class="ava" style="--tc:${teamColor(team)}${size?`;width:${size}px;height:${size}px;font-size:${Math.round(size*.36)}px`:''}" title="${esc(abbrOf(team))}">${esc(initials(name))}</span>`;
function pband(name, team, sub, chips=[]){ return `<div class="pband" style="--tc:${teamColor(team)}">${ava(name, team, 56)}<div><h2>${esc(name)}</h2><div class="hint">${sub}</div>${chips.length?`<div class="taglist" style="margin:6px 0 0">${chips.join(' ')}</div>`:''}</div></div>`; }

// ---------- theme + settings + navigation
function setTheme(t){ if(t) document.documentElement.setAttribute('data-theme', t); else document.documentElement.removeAttribute('data-theme'); try{ localStorage.setItem('lab-theme', t||''); }catch(e){} }
$('#theme-btn').addEventListener('click', ()=>{ const cur = document.documentElement.getAttribute('data-theme') || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'); setTheme(cur==='dark' ? 'light' : 'dark'); });
try{ const t=localStorage.getItem('lab-theme'); if(t) setTheme(t); }catch(e){}
$('#settings-btn').addEventListener('click', ()=>{ $('#settings').classList.toggle('open'); });
$('#settings-close').addEventListener('click', ()=>$('#settings').classList.remove('open'));
function goTab(k){ $$('[data-tab]').forEach(x=>x.setAttribute('aria-selected', x.dataset.tab===k)); $$('section[role="tabpanel"]').forEach(s=>s.classList.toggle('on', s.id==='tab-'+k)); try{localStorage.setItem('lab-tab',k);}catch(e){} window.scrollTo({top:0}); }
$$('[data-tab]').forEach(b=>b.addEventListener('click',()=>goTab(b.dataset.tab)));
$('#brand').addEventListener('click', e=>{ e.preventDefault(); goTab('dash'); });
$('#live-ind').addEventListener('click', ()=>{ if(window.refreshLive) window.refreshLive(); });

// ---------- global player search
const SEARCH_IDX = (()=>{ const m = new Map(); ROWS.filter(r=>r.season===CUR).forEach(r=>m.set('h'+r.mlbam_id, {id:r.mlbam_id, name:r.name, team:r.team_abbr, sub:`${r.elig||''} · ${Math.round(r.pts||0)} pts`, kind:'h'})); PROWS.filter(r=>r.season===CUR).forEach(r=>m.set('p'+r.mlbam_id, {id:r.mlbam_id, name:r.name, team:r.team, sub:`SP · ${Math.round(r.pts||0)} pts`, kind:'p'})); D.prospects.forEach(p=>{ if(!m.has('h'+p.mlbam_id)) m.set('h'+p.mlbam_id, {id:p.mlbam_id, name:p.name, team:(p.prospect_org||'').toUpperCase(), sub:`prospect · ${p.prospect_pos||''}`, kind:'h'}); }); return [...m.values()]; })();
const fold = s => (s||'').toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g,'');
(function buildSearch(){ const inp = $('#gsearch'), dd = $('#gsearch-dd'); let items=[];
  function render(){ dd.innerHTML = items.map((x,i)=>`<button role="option" data-i="${i}">${ava(x.name,x.team,26)}<span>${esc(x.name)}</span><small>${esc(abbrOf(x.team))} · ${esc(x.sub)}</small></button>`).join(''); dd.classList.toggle('open', items.length>0);
    dd.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{ const x=items[+b.dataset.i]; dd.classList.remove('open'); inp.value=''; (x.kind==='p'?openPitcherCard:openCard)(x.id); })); }
  inp.addEventListener('input', ()=>{ const q = fold(inp.value.trim()); if(q.length<2){ items=[]; render(); return; } items = SEARCH_IDX.filter(x=>fold(x.name).includes(q)).slice(0,8); render(); });
  inp.addEventListener('keydown', e=>{ if(e.key==='Enter' && items.length){ dd.querySelector('button').click(); } if(e.key==='Escape'){ items=[]; render(); } });
  document.addEventListener('click', e=>{ if(!e.target.closest('.search')) dd.classList.remove('open'); });
})();

// ---------- team switcher
(function buildTeamSel(){ const sel = $('#team-sel'); const owners = [...new Set([...Object.values(PROJ).map(p=>p.owner), ...Object.values(PPROJ).map(p=>p.owner)].filter(o=>o && !isFA(o)))].sort();
  sel.innerHTML = '<option value="">League view</option>' + owners.map(o=>`<option value="${esc(o)}">${esc(teamLabel(o))}</option>`).join(''); sel.value = owners.includes(MY) ? MY : '';
  if(!owners.includes(MY)) MY = '';
  sel.addEventListener('change', ()=>{ MY = sel.value; try{ localStorage.setItem('lab-team', MY); }catch(e){} const ko = $('#k-owner'); if(ko && MY && [...ko.options].some(o=>o.value===MY)) ko.value = MY; refreshAll(); renderStreamers(); renderPros(); });
})();

// ---------- dashboard
function renderDash(){
  const today = new Date(); const iso = d => d.toISOString().slice(0,10);
  const dates = [...new Set(SHD.map(r=>r.date))].sort(); const day = dates.find(d=>d>=iso(today)) || dates[0];
  $('#dash-sub').textContent = SD ? `Week of ${dayLabel(SD.meta.start)}. ${SD.meta.games} games, ${SD.meta.listed} with a posted probable so far.` : 'Season data through 2026.';
  $('#dash-asof').textContent = SD ? `Built ${SD.meta.generated}` : `Built ${D.meta.generated}`;
  const cur = ROWS.filter(r=>r.season===CUR && r.proj_pts!=null);
  const rowH = (r, val, sub2) => `<div class="row" data-h="${r.mlbam_id}">${ava(r.name, r.team_abbr||r.team, 32)}<div><div class="nm">${esc(r.name)}</div><div class="sub">${esc(sub2)}</div></div><div class="val">${val}</div></div>`;
  const rowP = (r, val, sub2) => `<div class="row" data-p="${r.mlbam_id}">${ava(r.name, r.team, 32)}<div><div class="nm">${esc(r.name)}</div><div class="sub">${esc(sub2)}</div></div><div class="val">${val}</div></div>`;
  let cards = '';
  if (!MY){ cards += `<div class="dcard w12"><h3>Pick your team <small>top bar</small></h3><p class="hint" style="font-size:13px">Choose your Fantrax team under "Viewing as" and this page becomes yours: lineup by day, your starts against the 10-start cap, your keeper board. Everything below is league-wide.</p></div>`; }
  if (SD && day && MY){
    const mine = SHD.filter(r=>r.owner===MY && r.active!==0 && r.date===day); const lu = bestLineup(mine); const tot = lu.reduce((a,[s,r])=>a+(r?r.exp_pts:0),0);
    cards += `<div class="dcard w6"><h3>Your lineup · ${dayLabel(day)} <small>${tot.toFixed(1)} expected pts</small></h3><div class="rowlist">${lu.map(([s,r])=>r?`<div class="row" data-h="${r.mlbam_id}">${ava(r.name,r.team,32)}<div><div class="nm"><span class="pill n" style="margin-right:6px">${s}</span>${esc(r.name)} ${lineupPill(r)}</div><div class="sub">${r.home?'vs':'@'} ${esc(r.opp)} · ${esc(r.opp_sp||'TBD')} ${r.opp_sp_throws||''}${r.opp_sp_source==='projected'?' (P)':''} · park ${r.park_runs??'–'}${r.temp_f!=null?` · ${Math.round(r.temp_f)}°`:''}</div></div><div class="val">${r.exp_pts.toFixed(1)}</div></div>`:`<div class="row"><span class="ava" style="--tc:var(--panel3)">${s}</span><div class="sub">open slot</div><div></div></div>`).join('')}</div><button class="go" data-go="streamers">Full week and bench →</button></div>`;
    const wx = [...new Map(SHD.filter(r=>r.owner===MY && r.active!==0 && r.date===day).map(r=>[r.gamePk, r])).values()];
    cards += `<div class="dcard w6"><h3>Your parks today <small>forecast at first pitch</small></h3><div class="todaystrip">${wx.map(r=>`<div class="wxchip"><b>${esc(abbrOf(r.team))} ${r.home?'vs':'@'} ${esc(abbrOf(r.opp))} ${stateChip(r.gamePk)}</b><span>${r.temp_f!=null?`${Math.round(r.temp_f)}° · wind ${r.wind_out>=0?'out':'in'} ${Math.abs(r.wind_out||0).toFixed(0)}`:(r.roof&&r.roof!=='Open'?'roof':'no forecast')}</span><small>${esc(r.local_start||'')}${r.precip_prob>=50?` · ${Math.round(r.precip_prob)}% rain`:''} · park ${r.park_runs??'–'}</small></div>`).join('')||'<span class="hint">no games</span>'}</div>
      ${(()=>{ const ms = SPS.filter(r=>r.owner===MY && r.active!==0 && r.sp_source!=='replaced'); const n = ms.length; return `<div class="stat" style="margin-top:12px"><b>${n}</b><span>of 10 starts this week from your rotation${n>10?' — you will have to bench one':''}</span></div>`; })()}</div>`;
    const faH = Object.values(SHD.filter(r=>isFA(r.owner)&&r.active!==0).reduce((a,r)=>{ (a[r.mlbam_id] ||= {...r, week:0, g:0}); a[r.mlbam_id].week += r.exp_pts; a[r.mlbam_id].g++; return a; },{})).sort((a,b)=>b.week-a.week).slice(0,6);
    cards += `<div class="dcard"><h3>Pickups: bats this week <small>free agents</small></h3><div class="rowlist">${faH.map(r=>rowH(r, `${r.week.toFixed(1)}<small>${r.g} games</small>`, `${r.elig||''} · ${esc(abbrOf(r.team))} · 30-day wOBA ${r.woba_30!=null?r.woba_30.toFixed(3):'–'}`)).join('')}</div><button class="go" data-go="streamers">All streamers →</button></div>`;
    const faP = SPS.filter(r=>isFA(r.owner)&&r.active!==0&&r.sp_source!=='replaced'&&goodRole(r)).sort((a,b)=>b.exp_pts-a.exp_pts).slice(0,6);
    cards += `<div class="dcard"><h3>Pickups: starts this week <small>free agents</small></h3><div class="rowlist">${faP.map(r=>rowP(r, `${r.exp_pts.toFixed(1)}<small>${dayLabel(r.date).split(',')[0]}${r.two_start?' · 2-start':''}</small>`, `${r.home?'vs':'@'} ${esc(r.opp)}${r.sp_source==='projected'?' (P)':''} · opp wOBA ${r.opp_woba_vs_hand?.toFixed(3)}${r.pl_tier?' · PL: '+esc(r.pl_tier):''}${r.cbs?' · CBS '+esc(r.cbs):''}`)).join('')}</div><button class="go" data-go="streamers">Pitcher slate →</button></div>`;
  }
  const kb = (MY ? cur.filter(r=>r.owner===MY && !r.pslot) : cur.filter(r=>!isFA(r.owner))).sort((a,b)=>(b.KSV_c??-999)-(a.KSV_c??-999)).slice(0,6);
  cards += `<div class="dcard"><h3>Keeper board <small>${MY?'2027, your five hitters':'2027, league-wide surplus value'}</small></h3><div class="rowlist">${kb.map((r,i)=>rowH(r, `${signed(r.KSV_c)}<small>KSV</small>`, `${i+1}. proj ${Math.round(r.proj_pts)} · ${r.tier_c||''}${MY?'':' · '+esc(r.owner)}${r.kept_2026_as&&MY===HOME_ABBR?' · kept 2026':''}`)).join('')}</div><button class="go" data-go="keepers">Boards and draft pool →</button></div>`;
  const bo = ROWS.filter(r=>r.season===CUR && r.BI!=null && r.PA>=150 && r.age<=30 && isFA(r.owner)).sort((a,b)=>b.BI-a.BI).slice(0,6);
  cards += `<div class="dcard"><h3>Breakout watch <small>free-agent bats, 2027 BI</small></h3><div class="rowlist">${bo.map(r=>rowH(r, `${r.BI.toFixed(0)}%<small>BI</small>`, `${r.elig||''} · age ${r.age} · ${Math.round(r.pts_c)} pts · xwOBA ${r.xwoba?.toFixed(3)??'–'}`)).join('')}</div><button class="go" data-go="breakouts">Full breakout board →</button></div>`;
  const asofT = SD ? new Date(SD.meta.asof+'T12:00:00') : today; const recent = ROWS.filter(r=>r.season===CUR && r.first_game && r.PA>=40 && r.PA<=250 && (asofT - new Date(r.first_game+'T12:00:00'))/864e5 <= 60).sort((a,b)=>(b.pts_pa_c||0)-(a.pts_pa_c||0)).slice(0,6);
  cards += `<div class="dcard"><h3>Just called up <small>first 2026 game in the last 60 days</small></h3><div class="rowlist">${recent.map(r=>rowH(r, `${(r.pts_pa_c||0).toFixed(2)}<small>pts/PA · ${r.PA} PA</small>`, `${r.elig||''} · ${esc(abbrOf(r.team_abbr))} · up ${esc(r.first_game)} · pace ${r.pa_pace_162?Math.round(r.pa_pace_162):'–'} PA · xwOBA ${r.xwoba?.toFixed(3)??'–'} · ${isFA(r.owner)?'free agent':esc(r.owner)}`)).join('')||'<span class="hint">none</span>'}</div><button class="go" data-go="prospects">Prospect cards →</button></div>`;
  const pb = PROWS.filter(r=>r.season===CUR && r.BI!=null && r.GS>=10 && isFA(r.owner)).sort((a,b)=>b.BI-a.BI).slice(0,6);
  cards += `<div class="dcard"><h3>Arms to watch <small>free-agent SP, 2027 BI</small></h3><div class="rowlist">${pb.map(r=>rowP(r, `${r.BI.toFixed(0)}%<small>BI</small>`, `${r.GS} GS · ${(r.pts_gs_c??0).toFixed(1)}/GS · Pitching+ ${r.pitching_plus?.toFixed(0)??'–'}`)).join('')}</div><button class="go" data-go="pitchers">Pitcher lab →</button></div>`;
  $('#dash').innerHTML = cards;
  $$('#dash .row[data-h]').forEach(el=>el.addEventListener('click',()=>openCard(+el.dataset.h)));
  $$('#dash .row[data-p]').forEach(el=>el.addEventListener('click',()=>openPitcherCard(+el.dataset.p)));
  $$('#dash .go').forEach(b=>b.addEventListener('click',()=>goTab(b.dataset.go)));
}
'''


def main(argv=None):
    v3 = (C.OUT / "v3" / "explorer_template.html").read_text(encoding="utf-8")
    i = v3.index('<section role="tabpanel" id="tab-explore"'); j = v3.index('</div>\n\n<div class="overlay"')
    sections = v3[i:j]
    js = v3[v3.index('<script id="data"'):]
    # explorer section: the first section carries class="on"; dashboard is the landing page now
    sections = sections.replace('<section role="tabpanel" id="tab-explore" class="on">', '<section role="tabpanel" id="tab-explore">\n  <div class="pagehead"><div><h2>Hitters</h2><p class="lede">Every hitter-season since 2021 scored in your league\'s points. Filter, add stat ranges, pick columns; click a name for the player page.</p></div></div>')
    # page heads for other sections: turn the existing h2 into pagehead h2 (already h2) — fine as is
    # JS edits: navigation/boot, player page headers
    js = js.replace("$$('nav.tabs button').forEach(b=>b.addEventListener('click',()=>{ $$('nav.tabs button').forEach(x=>x.setAttribute('aria-selected',x===b)); $$('section[role=\"tabpanel\"]').forEach(s=>s.classList.toggle('on', s.id==='tab-'+b.dataset.tab)); try{localStorage.setItem('lab-tab',b.dataset.tab);}catch(e){} }));\n", "")
    js = js.replace("try{ const t=localStorage.getItem('lab-tab'); if(t) $(`nav.tabs button[data-tab=\"${t}\"]`)?.click(); }catch(e){}", "renderDash(); try{ const t=localStorage.getItem('lab-tab'); if(t && $(`[data-tab=\"${t}\"]`)) goTab(t); }catch(e){}")
    assert "renderDash();" in js, "boot hook not found"
    js = js.replace("function refreshAll(){ rescore(); prescore(); buildPct(); renderExplorer(); renderBreak(); renderPitch(); renderKeepers(); renderDrafts(); renderFormulaScoring(); }",
                    "function refreshAll(){ rescore(); prescore(); buildPct(); renderExplorer(); renderBreak(); renderPitch(); renderKeepers(); renderDrafts(); renderFormulaScoring(); renderDash(); }")
    # hitter page header
    old = "  const head = sub ? `<h3 style=\"font-size:24px;margin-top:22px;border-top:2px solid var(--line);padding-top:14px\">As a hitter</h3>` : `<h2>${esc(last.name)}</h2>`;\n  return `${head}<div class=\"hint\">${esc(last.elig||'')} · ${esc(last.team_abbr||'')} · age ${last.age} · owner ${esc(last.owner||'–')} · ${esc(last.prospect_status||'')}${last.pipeline_rank?' · Pipeline #'+last.pipeline_rank:''}${p?.keep_tier?' · keeper: '+p.keep_tier:''}${keptTag}</div>"
    new = "  const subline = `${esc(last.elig||'')} · ${esc(last.team_abbr||'')} · bats ${esc(last.bats||'?')} · age ${last.age} · owner ${esc(last.owner||'–')}${last.prospect_status?' · '+esc(last.prospect_status):''}${last.pipeline_rank?' · Pipeline #'+last.pipeline_rank:''}`;\n  const chips = [p?.keep_tier?`<span class=\"pill\">keeper: ${p.keep_tier}</span>`:'', p?.kept_2026_as?`<span class=\"pill\">your 2026 keeper (${p.kept_2026_as})</span>`:''].filter(Boolean);\n  const head = sub ? `<h3 style=\"font-size:24px;margin-top:22px;border-top:2px solid var(--line);padding-top:14px\">As a hitter</h3><div class=\"hint\">${subline}</div>` : pband(last.name, last.team_abbr, subline, chips);\n  return `${head}<div class=\"card-body\">"
    assert old in js, "hitter head not found"; js = js.replace(old, new)
    js = js.replace("${p ? `<h3>2027 projection</h3><p class=\"hint\">rate ${p.proj_rate?.toFixed(3)} pts/PA (model ${p.proj_rate_raw?.toFixed(3)}, aging step ${(p.age_step??0)>=0?'+':''}${(p.age_step??0).toFixed(3)}), durability ×${p.durability}, projected PA ${p.proj_PA}. BI ${p.BI} = chance 2027 is a new career level.</p>` : ''}`;\n}",
                    "${p ? `<h3>2027 projection</h3><p class=\"hint\">rate ${p.proj_rate?.toFixed(3)} pts/PA (model ${p.proj_rate_raw?.toFixed(3)}, aging step ${(p.age_step??0)>=0?'+':''}${(p.age_step??0).toFixed(3)}), durability ×${p.durability}, projected PA ${p.proj_PA}. BI ${p.BI} = chance 2027 is a new career level.</p>` : ''}</div>`;\n}")
    # pitcher page header
    old = "  const head = sub ? `<h3 style=\"font-size:24px;margin-top:22px;border-top:2px solid var(--line);padding-top:14px\">As a pitcher</h3>` : `<h2>${esc(last.name)}</h2>`;\n  return `${head}<div class=\"hint\">SP · ${esc(last.throws||'')}HP · ${esc(last.team||'')} · age ${last.age} · owner ${esc(last.owner||'–')}${p?.keep_tier?' · keeper: '+p.keep_tier:''}${keptTag}</div>"
    new = "  const subline = `SP · ${esc(last.throws||'')}HP · ${esc(last.team||'')} · age ${last.age} · owner ${esc(last.owner||'–')}`;\n  const chips = [p?.keep_tier?`<span class=\"pill\">keeper: ${p.keep_tier}</span>`:'', p?.kept_2026_as?`<span class=\"pill\">your 2026 keeper (${p.kept_2026_as})</span>`:''].filter(Boolean);\n  const head = sub ? `<h3 style=\"font-size:24px;margin-top:22px;border-top:2px solid var(--line);padding-top:14px\">As a pitcher</h3><div class=\"hint\">${subline}</div>` : pband(last.name, last.team, subline, chips);\n  return `${head}<div class=\"card-body\">"
    assert old in js, "pitcher head not found"; js = js.replace(old, new)
    js = js.replace("BI ${p.BI} = chance 2027 is a new career level in points per start.</p>` : ''}`;\n}", "BI ${p.BI} = chance 2027 is a new career level in points per start.</p>` : ''}</div>`;\n}")
    # prospect header
    old = "function prospectHeader(p){\n  return `<h2>${esc(p.name)}</h2><div class=\"hint\">${esc(p.prospect_pos||'')} · ${esc((p.prospect_org||'').toUpperCase())} · age ${p.age} · owner ${esc(p.owner||'–')} · ${esc(p.prospect_status||'')}${p.pipeline_rank?' · Pipeline #'+p.pipeline_rank:''}${p.rookie_eligible?' · rookie-eligible':''}</div>`;\n}"
    new = "function prospectHeader(p){\n  return pband(p.name, (p.prospect_org||'').toUpperCase(), `${esc(p.prospect_pos||'')} · ${esc((p.prospect_org||'').toUpperCase())} · age ${p.age} · owner ${esc(p.owner||'–')} · ${esc(p.prospect_status||'')}${p.pipeline_rank?' · Pipeline #'+p.pipeline_rank:''}${p.rookie_eligible?' · rookie-eligible':''}`);\n}"
    assert old in js, "prospect head not found"; js = js.replace(old, new)
    # prospect section body wrapper
    js = js.replace("  return `${kpis}${pas}${ltable}${scHtml}<p class=\"hint\">Percentiles in the Statcast block", "  return `<div class=\"card-body\">${kpis}${pas}${ltable}${scHtml}<p class=\"hint\">Percentiles in the Statcast block")
    js = js.replace("minor-league lines are against every hitter with 100+ PA at that level.</p>`;\n}", "minor-league lines are against every hitter with 100+ PA at that level.</p></div>`;\n}")
    # inject extras before the boot line
    marker = "// ---------- tabs + boot"
    assert marker in js; js = js.replace(marker, JS_EXTRA + "\n" + marker)
    html = shell(sections) + js
    out = C.OUT / "v4"; out.mkdir(exist_ok=True); (out / "explorer_template.html").write_text(html, encoding="utf-8")
    print("wrote", out / "explorer_template.html", len(html))
    return 0


if __name__ == "__main__":
    sys.exit(main())
