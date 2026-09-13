/* Live layer for the standalone site. Runs after the main app boots.
   Pulls, straight from the browser (no server): MLB schedule with posted probables, game states and scores,
   today's lineups (once posted) and Open-Meteo first-pitch weather, then re-renders the Streamers tab and dashboard.
   Everything else (projections, Statcast, keeper values, ownership) comes from the daily build. */
(function(){
  if (typeof SD === 'undefined' || !SD) return;
  const API = 'https://statsapi.mlb.com/api/v1';
  const LG_XWOBA = 0.312;
  const $live = document.getElementById('live-ind');
  const setInd = (txt, cls) => { if($live){ $live.textContent = txt; $live.className = 'liveind ' + (cls||''); } };
  const localDate = () => { const d = new Date(); return new Date(d.getTime() - d.getTimezoneOffset()*60000).toISOString().slice(0,10); };
  const clip = (x,a,b) => Math.max(a, Math.min(b, x));
  const pitcherInfo = id => { const h = PBY_ID[id]; if(!h) return null; const r = h[h.length-1]; const bf = (r.IP||0) * 4.3; const xw = ((r.xwoba ?? LG_XWOBA) * bf + LG_XWOBA * 300) / (bf + 300); return {name:r.name, throws:r.throws, xwoba:xw, k:r.k_percent, pp:r.pitching_plus, pts_gs:r.pts_gs_c ?? r.pts_gs, GS:r.GS}; };
  const spFactor = xw => Math.pow(clip(xw / LG_XWOBA, 0.75, 1.3), 0.6);
  const byGame = {}; SGM.forEach(g => (byGame[g.gamePk] ||= []).push(g));
  const gameState = {};

  async function getJSON(u){ const ctl = new AbortController(); const tm = setTimeout(()=>ctl.abort(), 15000); try { const r = await fetch(u, {cache:'no-store', signal: ctl.signal}); if(!r.ok) throw new Error(r.status); return await r.json(); } finally { clearTimeout(tm); } }

  async function refreshSchedule(){
    const j = await getJSON(`${API}/schedule?sportId=1&startDate=${SD.meta.start}&endDate=${SD.meta.end}&hydrate=probablePitcher,linescore`);
    let changed = 0;
    for (const d of j.dates||[]) for (const g of d.games){
      const st = g.status?.detailedState || ''; const ls = g.linescore || {};
      gameState[g.gamePk] = { state: st, inning: ls.currentInning, half: ls.inningState, home: ls.teams?.home?.runs, away: ls.teams?.away?.runs, live: /In Progress|Delayed|Suspended/.test(st), final: /Final|Completed|Game Over/.test(st), ppd: /Postponed|Cancelled/.test(st) };
      for (const side of ['home','away']){
        const t = g.teams[side]; const pp = t.probablePitcher; if(!pp) continue;
        const teamName = t.team.name; const oppSide = side==='home'?'away':'home';
        // games table: this team's starter
        (byGame[g.gamePk]||[]).forEach(row => {
          if (row.team===teamName && row.sp_name!==pp.fullName){ row.sp_name = pp.fullName; row.sp_source = 'listed'; changed++; }
          if (row.team===g.teams[oppSide].team.name && row.opp_sp_name!==pp.fullName){ row.opp_sp_name = pp.fullName; row.opp_sp_source = 'listed'; }
        });
        // hitters of the OTHER team face this pitcher
        const info = pitcherInfo(pp.id);
        SHD.filter(r => r.gamePk===g.gamePk && r.team===g.teams[oppSide].team.name).forEach(r => {
          if (r.opp_sp===pp.fullName && r.opp_sp_source==='listed') return;
          r.opp_sp = pp.fullName; r.opp_sp_source = 'listed';
          if (info){ r.opp_sp_throws = info.throws || r.opp_sp_throws; const f = spFactor(info.xwoba); if (r.f_sp){ r.mult = +(r.mult / r.f_sp * f).toFixed(3); r.exp_pts = +(r.base_rate * r.pa_g * r.mult).toFixed(2); } r.f_sp = +f.toFixed(3); r.sp_xwoba = +info.xwoba.toFixed(3); r.sp_k = info.k; r.sp_pitching_plus = info.pp; }
        });
        // pitcher starts: replace a projected starter with the posted one
        const rows = SPS.filter(r => r.gamePk===g.gamePk && r.team===teamName);
        const have = rows.find(r => r.name===pp.fullName);
        if (!have){
          const proj = rows.find(r => r.sp_source==='projected');
          if (proj){
            proj.sp_source = 'replaced';
            if (info){ const base = ((info.pts_gs||7) * Math.min(info.GS||0, 20) + 7 * 10) / (Math.min(info.GS||0, 20) + 10);
              const owner = (PPROJ[pp.id]||{}).owner || 'FA';
              SPS.push({...proj, mlbam_id: pp.id, name: pp.fullName, throws: info.throws, owner, sp_source:'listed', base_gs:+base.toFixed(2), pts_gs_2026: info.pts_gs, GS: info.GS, pitching_plus: info.pp, k_percent: info.k, role:'starter', n_starts: info.GS, exp_pts: +(base * proj.f_opp * proj.f_park * proj.f_wx * proj.f_home + (proj.k_bonus||0)).toFixed(2), two_start: false, pl_tier: null, cbs: null, active: 1, live: true});
            }
          }
        } else if (have.sp_source!=='listed'){ have.sp_source = 'listed'; }
      }
    }
    // recompute two-start flags and week totals for pitchers
    const cnt = {}; SPS.filter(r=>r.sp_source!=='replaced').forEach(r => cnt[r.mlbam_id] = (cnt[r.mlbam_id]||0)+1);
    const wk = {}; SPS.filter(r=>r.sp_source!=='replaced').forEach(r => wk[r.mlbam_id] = (wk[r.mlbam_id]||0)+r.exp_pts);
    SPS.forEach(r => { r.two_start = (cnt[r.mlbam_id]||0) >= 2; r.week_pts = +(wk[r.mlbam_id]||0).toFixed(1); });
    return changed;
  }

  async function refreshLineups(){
    const today = localDate(); const games = [...new Set(SHD.filter(r=>r.date===today).map(r=>r.gamePk))];
    let posted = 0;
    await Promise.all(games.map(async pk => {
      try {
        const b = await getJSON(`${API}/game/${pk}/boxscore`);
        for (const side of ['home','away']){
          const order = b.teams?.[side]?.battingOrder || []; if(!order.length) continue; posted++;
          const teamName = b.teams[side].team.name; const spot = {}; order.forEach((id,i)=>spot[id]=i+1);
          SHD.filter(r => r.gamePk===pk && r.team===teamName).forEach(r => { r.lineup = spot[r.mlbam_id] || 0; });
        }
      } catch(e){}
    }));
    return posted;
  }

  async function refreshWeather(){
    const today = localDate(); const ven = Object.fromEntries((SD.venues||[]).map(v=>[v.venue_id, v]));
    const todays = SGM.filter(g => g.date===today); const byVenue = {};
    todays.forEach(g => { const r = SHD.find(x=>x.gamePk===g.gamePk); if(r) (byVenue[r.venue] ||= []).push(g.gamePk); });
    for (const [vname, pks] of Object.entries(byVenue)){
      const v = Object.values(ven).find(x=>x.venue===vname); if(!v || v.lat==null) continue;
      try {
        const j = await getJSON(`https://api.open-meteo.com/v1/forecast?latitude=${v.lat}&longitude=${v.lon}&hourly=temperature_2m,wind_speed_10m,wind_direction_10m,precipitation_probability&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=${encodeURIComponent(v.tz||'America/New_York')}&start_date=${today}&end_date=${today}`);
        const h = j.hourly; const rows = SHD.filter(r => pks.includes(r.gamePk));
        rows.forEach(r => { if(!r.local_start) return; const m = r.local_start.match(/(\d+):(\d+) (AM|PM)/); if(!m) return; let hr = +m[1]%12 + (m[3]==='PM'?12:0); const i = h.time.findIndex(t=>t.endsWith(`T${String(hr).padStart(2,'0')}:00`)); if(i<0) return;
          const temp = (h.temperature_2m[i]+ (h.temperature_2m[i+1]??h.temperature_2m[i]) + (h.temperature_2m[i+2]??h.temperature_2m[i]))/3; const ws = h.wind_speed_10m[i]; const wd = h.wind_direction_10m[i];
          const roofClosed = v.roof && /dome|retractable|fixed/i.test(v.roof);
          const toward = (wd+180)%360; const comp = Math.cos((toward - (v.azimuth||0))*Math.PI/180) * ws;
          let tm = 1; if(!roofClosed){ tm = clip(1 + 0.003*(temp-70), 0.94, 1.06) * clip(1 + 0.004*comp, 0.92, 1.08); }
          if (r.f_wx){ r.mult = +(r.mult / r.f_wx * tm).toFixed(3); r.exp_pts = +(r.base_rate * r.pa_g * r.mult).toFixed(2); }
          r.f_wx = +tm.toFixed(3); r.temp_f = Math.round(temp); r.wind_mph = Math.round(ws); r.wind_out = +comp.toFixed(1); r.precip_prob = Math.max(...h.precipitation_probability.slice(i, i+3)); });
      } catch(e){}
    }
  }

  window.gameState = gameState;
  window.refreshLive = async function(){
    setInd('LIVE · updating…', 'busy');
    try {
      const ch = await refreshSchedule(); const posted = await refreshLineups(); await refreshWeather();
      if (typeof renderStreamers==='function') renderStreamers(); if (typeof renderDash==='function') renderDash(); if (typeof renderToday==='function') renderToday();
      const t = new Date().toLocaleTimeString([], {hour:'numeric', minute:'2-digit'});
      setInd(`LIVE · ${t}${posted?` · ${posted} lineups posted`:''}`, 'ok');
    } catch(e){ setInd('LIVE · offline', 'err'); }
  };
  window.refreshLive();
  setInterval(() => { if (document.visibilityState==='visible') window.refreshLive(); }, 10*60*1000);
  document.addEventListener('visibilitychange', () => { if (document.visibilityState==='visible') window.refreshLive(); });
})();
