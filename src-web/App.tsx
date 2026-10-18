import { useEffect, useMemo, useState } from 'react';
import {
  ArrowDownRight, ArrowUpRight, Cloud, CloudDrizzle, CloudRain, Compass, ExternalLink,
  MapPin, RefreshCw, Sun, Thermometer, Wind,
} from 'lucide-react';
import { classifyWeather, predictMaximumTemperature, predictRainfall } from './inference';
import type { BrowserModel, WeatherDataset, WeatherDay } from './types';

type Range = '90d' | '1y' | '5y' | 'all';

const fmtDate = (date: string, options: Intl.DateTimeFormatOptions = { day: 'numeric', month: 'short', year: 'numeric' }) =>
  new Intl.DateTimeFormat('en-AU', { timeZone: 'UTC', ...options }).format(new Date(`${date}T00:00:00Z`));
const tomorrow = (date: string) => {
  const next = new Date(`${date}T00:00:00Z`);
  next.setUTCDate(next.getUTCDate() + 1);
  return next.toISOString().slice(0, 10);
};
const shortNumber = (n: number) => new Intl.NumberFormat('en-AU', { maximumFractionDigits: 1 }).format(n);

function TemperatureChart({ rows }: { rows: WeatherDay[] }) {
  const sampled = useMemo(() => rows.filter((_, index) => index % Math.max(1, Math.ceil(rows.length / 280)) === 0 || index === rows.length - 1), [rows]);
  if (!sampled.length) return null;
  const values = sampled.flatMap((row) => [row.tmax, row.tmin]);
  const low = Math.floor(Math.min(...values) / 5) * 5;
  const high = Math.ceil(Math.max(...values) / 5) * 5;
  const x = (i: number) => 18 + (i / Math.max(1, sampled.length - 1)) * 964;
  const y = (v: number) => 15 + ((high - v) / Math.max(1, high - low)) * 180;
  const pathFor = (key: 'tmax' | 'tmin') => sampled.map((row, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(row[key]).toFixed(1)}`).join(' ');
  const fillFor = (key: 'tmax' | 'tmin') => `${pathFor(key)} L${x(sampled.length - 1)},205 L${x(0)},205 Z`;
  return <div className="chart-wrap">
    <svg className="temperature-chart" viewBox="0 0 1000 225" role="img" aria-label="Historical daily maximum and minimum temperature chart">
      {[0, 1, 2, 3].map((i) => <g key={i}><line x1="18" x2="982" y1={15 + i * 60} y2={15 + i * 60} className="chart-grid" /><text x="0" y={19 + i * 60} className="chart-y-label">{high - Math.round((high - low) * i / 3)}°</text></g>)}
      <path d={fillFor('tmax')} className="temp-fill temp-fill-high" />
      <path d={fillFor('tmin')} className="temp-fill temp-fill-low" />
      <path d={pathFor('tmax')} className="chart-line chart-line-high" />
      <path d={pathFor('tmin')} className="chart-line chart-line-low" />
    </svg>
    <div className="chart-dates"><span>{fmtDate(sampled[0].date, { month: 'short', year: 'numeric' })}</span><span>{fmtDate(sampled[Math.floor(sampled.length / 2)].date, { month: 'short', year: 'numeric' })}</span><span>{fmtDate(sampled.at(-1)!.date, { month: 'short', year: 'numeric' })}</span></div>
  </div>;
}

function RainChart({ rows }: { rows: WeatherDay[] }) {
  const sampled = useMemo(() => rows.filter((_, index) => index % Math.max(1, Math.ceil(rows.length / 120)) === 0 || index === rows.length - 1), [rows]);
  const maximum = Math.max(1, ...sampled.map((row) => row.prcp));
  return <div className="rain-bars" role="img" aria-label="Daily precipitation history">
    {sampled.map((row, index) => <span key={`${row.date}-${index}`} title={`${fmtDate(row.date)} · ${row.prcp.toFixed(1)} mm`} style={{ height: `${Math.max(3, (row.prcp / maximum) * 100)}%` }} />)}
  </div>;
}

function App() {
  const [dataset, setDataset] = useState<WeatherDataset | null>(null);
  const [model, setModel] = useState<BrowserModel | null>(null);
  const [error, setError] = useState('');
  const [range, setRange] = useState<Range>('1y');
  const [minimum, setMinimum] = useState(10);
  const [rainInput, setRainInput] = useState(0);
  const [selectedTab, setSelectedTab] = useState<'overview' | 'history' | 'groups'>('overview');

  useEffect(() => {
    let active = true;
    Promise.all([
      fetch('/data/weather.json').then((response) => { if (!response.ok) throw new Error('Weather history could not be loaded.'); return response.json() as Promise<WeatherDataset>; }),
      fetch('/models/inference.json').then((response) => { if (!response.ok) throw new Error('The trained model files could not be loaded.'); return response.json() as Promise<BrowserModel>; }),
    ]).then(([weather, trainedModel]) => {
      if (!active) return;
      setDataset(weather);
      setModel(trainedModel);
      const latest = weather.records.at(-1);
      if (latest) { setMinimum(latest.tmin); setRainInput(latest.prcp); }
    }).catch((cause: unknown) => active && setError(cause instanceof Error ? cause.message : 'The dashboard could not be loaded.'));
    return () => { active = false; };
  }, []);

  const latest = dataset?.records.at(-1);
  const rainForecast = dataset && model ? predictRainfall(dataset.records, dataset, model) : null;
  const maxForecast = model ? predictMaximumTemperature(minimum, rainInput, model) : null;
  const forecastGroup = model && maxForecast !== null ? classifyWeather({ tmin: minimum, tmax: maxForecast, prcp: rainInput }, model) : null;
  const latestGroup = model && latest ? classifyWeather(latest, model) : null;
  const rainfallDays = dataset?.records.filter((day) => day.prcp > 0).length ?? 0;
  const lastYear = dataset?.records.slice(-365) ?? [];
  const recentRainMean = lastYear.length ? lastYear.reduce((sum, day) => sum + day.prcp, 0) / lastYear.length : 0;
  const visibleRows = useMemo(() => {
    if (!dataset) return [];
    if (range === 'all') return dataset.records;
    const days = range === '90d' ? 90 : range === '1y' ? 365 : 365 * 5;
    return dataset.records.slice(-days);
  }, [dataset, range]);

  if (error) return <main className="load-screen"><div className="load-card"><CloudRain size={28} /><h1>Weather data didn’t load</h1><p>{error}</p><button className="button-primary" onClick={() => location.reload()}><RefreshCw size={15} /> Try again</button></div></main>;
  if (!dataset || !model || !latest) return <main className="load-screen"><div className="loading-mark"><Cloud size={28} /><span>Preparing the station record…</span></div></main>;

  return <div className="min-h-screen bg-[#f5f6f2] text-[#18332e]">
    <header className="topbar">
      <a className="brand" href="#top" aria-label="Melbourne Weather Lab home"><span className="brand-mark"><Cloud size={19} strokeWidth={2.2} /></span><span>Weather<span className="brand-light">lab</span></span></a>
      <div className="topbar-right"><span className="data-live"><i /> NOAA DATA</span><a href="https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/ASN00086282.dly" target="_blank" rel="noreferrer" className="source-link">Source <ExternalLink size={13} /></a></div>
    </header>

    <main id="top" className="page-shell">
      <section className="welcome-row">
        <div><div className="location-line"><MapPin size={14} /> MELBOURNE AIRPORT <span>·</span> AUSTRALIA</div><h1>Weather, in context.</h1><p className="welcome-copy">A local outlook shaped by <strong>{(dataset.records.length / 1000).toFixed(1)}k days</strong> of station history.</p></div>
        <div className="updated-stamp"><span>LAST OBSERVATION</span><strong>{fmtDate(latest.date)}</strong><small>GHCN-Daily · {dataset.station}</small></div>
      </section>

      <nav className="tab-row" aria-label="Dashboard views">
        {(['overview', 'history', 'groups'] as const).map((tab) => <button key={tab} onClick={() => setSelectedTab(tab)} className={`tab-button ${selectedTab === tab ? 'tab-active' : ''}`}>{tab === 'overview' ? 'Overview' : tab === 'history' ? 'Climate history' : 'Weather groups'}</button>)}
        <span className="range-note"><Compass size={14} /> Southern hemisphere seasons</span>
      </nav>

      {selectedTab === 'overview' && <>
        <section className="overview-grid">
          <article className="today-panel">
            <div className="panel-top"><div><span className="eyebrow">LATEST STATION CONDITIONS</span><h2>{fmtDate(latest.date, { weekday: 'long', day: 'numeric', month: 'long' })}</h2></div><span className="weather-symbol"><Sun size={24} strokeWidth={1.5} /></span></div>
            <div className="temperature-reading"><strong>{shortNumber(latest.tmax)}°</strong><span>/{shortNumber(latest.tmin)}°</span><span className="degrees-unit">C</span></div>
            <div className="condition-line"><span className="weather-dot" /> {latestGroup?.label ?? 'Observed conditions'} <span className="condition-divider">/</span> Daily observation</div>
            <div className="today-foot"><span><CloudDrizzle size={15} /> {latest.prcp.toFixed(1)} mm rain</span><span><Thermometer size={15} /> Max / min temperature</span></div>
          </article>

          <article className="forecast-panel">
            <div className="panel-top"><div><span className="eyebrow">MODEL FORECAST · LSTM</span><h2>Tomorrow’s rainfall</h2></div><span className="model-chip">30 DAY WINDOW</span></div>
            <div className="forecast-reading"><strong>{rainForecast?.toFixed(1) ?? '—'}</strong><span>mm</span></div>
            <p className="forecast-date">For {fmtDate(tomorrow(latest.date))}</p>
            <div className="forecast-context"><CloudRain size={17} /><span>Estimated from the previous 30 observed days. This experimental model predicts rainfall amount.</span></div>
          </article>
        </section>

        <section className="quick-facts" aria-label="Station summary">
          <div><span className="fact-label">LONG-TERM HIGH</span><strong>{Math.max(...dataset.records.map((day) => day.tmax)).toFixed(1)}<small>°C</small></strong><span className="fact-detail"><ArrowUpRight size={13} /> Observed maximum</span></div>
          <div><span className="fact-label">LONG-TERM LOW</span><strong>{Math.min(...dataset.records.map((day) => day.tmin)).toFixed(1)}<small>°C</small></strong><span className="fact-detail"><ArrowDownRight size={13} /> Observed minimum</span></div>
          <div><span className="fact-label">RAIN DAYS</span><strong>{rainfallDays.toLocaleString('en-AU')}<small> days</small></strong><span className="fact-detail"><CloudRain size={13} /> Since {dataset.startYear}</span></div>
          <div><span className="fact-label">MEAN DAILY RAIN</span><strong>{recentRainMean.toFixed(1)}<small> mm</small></strong><span className="fact-detail"><Wind size={13} /> Rolling 12 months</span></div>
        </section>

        <section className="content-grid">
          <article className="surface history-surface">
            <div className="section-heading"><div><span className="eyebrow">OBSERVATION RECORD</span><h2>Temperature over time</h2><p>Daily highs and lows reveal Melbourne’s seasonal rhythm.</p></div><div className="range-select" aria-label="Select history range">{(['90d', '1y', '5y', 'all'] as const).map((value) => <button key={value} onClick={() => setRange(value)} className={range === value ? 'range-active' : ''}>{value === 'all' ? 'All' : value.toUpperCase()}</button>)}</div></div>
            <TemperatureChart rows={visibleRows} />
            <div className="chart-legend"><span><i className="legend-high" /> Maximum temperature</span><span><i className="legend-low" /> Minimum temperature</span><span className="units-label">°C · {visibleRows.length.toLocaleString('en-AU')} daily records</span></div>
          </article>
          <article className="surface rainfall-surface">
            <div className="section-heading"><div><span className="eyebrow">RAINFALL HISTORY</span><h2>Rain, day by day</h2><p>Tap a bar to inspect the station reading.</p></div></div>
            <RainChart rows={visibleRows} />
            <div className="rainfall-foot"><span>Dry days sit at the baseline.</span><strong>{visibleRows.reduce((sum, day) => sum + day.prcp, 0).toFixed(0)} mm <small>in view</small></strong></div>
          </article>
        </section>

        <section className="prediction-section">
          <div className="prediction-copy"><span className="eyebrow">TRY A SCENARIO</span><h2>What might tomorrow feel like?</h2><p>Adjust the inputs to explore the Linear Regression estimate and find the closest weather group.</p><div className="model-method"><span>01</span><div><strong>Linear Regression</strong><small>Maximum temperature estimate</small></div><span>+</span><div><strong>K-Means</strong><small>Nearest condition cluster</small></div></div></div>
          <div className="prediction-controls">
            <label htmlFor="minimum-input">Expected minimum temperature <span>{minimum.toFixed(1)} °C</span></label>
            <input id="minimum-input" type="range" min="-5" max="30" step="0.5" value={minimum} onChange={(event) => setMinimum(Number(event.target.value))} />
            <div className="range-ends"><span>−5 °C</span><span>30 °C</span></div>
            <label htmlFor="rain-input">Expected precipitation <span>{rainInput.toFixed(1)} mm</span></label>
            <input id="rain-input" type="range" min="0" max="60" step="0.5" value={Math.min(60, rainInput)} onChange={(event) => setRainInput(Number(event.target.value))} />
            <div className="range-ends"><span>0 mm</span><span>60+ mm</span></div>
          </div>
          <div className="prediction-result"><span className="eyebrow">MODEL ESTIMATE</span><div className="result-temperature">{maxForecast?.toFixed(1)}<small>°C</small></div><span className="result-label">Estimated maximum</span><div className="group-result"><span>Closest weather group</span><strong>{forecastGroup?.label}</strong><small>{forecastGroup?.profile.tmin.toFixed(1)}° / {forecastGroup?.profile.tmax.toFixed(1)}° mean · {forecastGroup?.profile.prcp.toFixed(1)} mm avg rain</small></div></div>
        </section>
        <p className="honesty-note">Research demo only. Historical records and model estimates are not an official forecast or severe weather warning.</p>
      </>}

      {selectedTab === 'history' && <section className="single-view"><div className="surface full-history"><div className="section-heading"><div><span className="eyebrow">CLIMATE ARCHIVE · {dataset.startYear}—{dataset.endYear}</span><h2>Melbourne’s temperature record</h2><p>Scroll across the full daily record, or narrow the view with the date range.</p></div><div className="range-select">{(['90d', '1y', '5y', 'all'] as const).map((value) => <button key={value} onClick={() => setRange(value)} className={range === value ? 'range-active' : ''}>{value === 'all' ? 'All years' : value.toUpperCase()}</button>)}</div></div><TemperatureChart rows={visibleRows} /><div className="chart-legend"><span><i className="legend-high" /> Daily maximum temperature</span><span><i className="legend-low" /> Daily minimum temperature</span><span className="units-label">°C · NOAA GHCN-Daily</span></div><div className="full-rain"><div><span className="eyebrow">DAILY PRECIPITATION</span><h2>Rainfall in the same period</h2></div><RainChart rows={visibleRows} /></div></div></section>}

      {selectedTab === 'groups' && <section className="single-view groups-view"><div className="groups-intro"><div><span className="eyebrow">UNSUPERVISED LEARNING</span><h2>Four patterns in the weather</h2><p>K-Means finds recurring combinations of minimum temperature, maximum temperature and rainfall. The group names below follow average maximum temperature.</p></div><div className="latest-cluster"><span className="eyebrow">LATEST OBSERVATION</span><strong>{latestGroup?.label}</strong><small>{fmtDate(latest.date)}</small></div></div><div className="group-table"><div className="group-table-head"><span>WEATHER GROUP</span><span>MEAN MIN</span><span>MEAN MAX</span><span>MEAN RAIN</span></div>{model.clusters.groups.map((group, index) => <div className={`group-row ${latestGroup?.id === group.id ? 'group-row-current' : ''}`} key={group.id}><span><i className={`cluster-swatch swatch-${index}`} />{group.label}{latestGroup?.id === group.id && <small> latest</small>}</span><strong>{group.profile.tmin.toFixed(1)}°C</strong><strong>{group.profile.tmax.toFixed(1)}°C</strong><strong>{group.profile.prcp.toFixed(1)} mm</strong></div>)}</div><div className="group-footnote"><Compass size={16} /><p>Cluster IDs are statistical groupings, not official weather categories. The standardized variables have equal footing in the distance calculation; compare the mean values to understand each group.</p></div></section>}

      <footer className="page-footer"><div><span className="footer-brand">Weatherlab</span><span>Built on real daily observations from Melbourne Airport.</span></div><div><a href="https://www.ncei.noaa.gov/pub/data/ghcn/daily/" target="_blank" rel="noreferrer">NOAA NCEI GHCN-Daily <ExternalLink size={12} /></a><span>Educational machine learning project · {dataset.startYear}–{dataset.endYear}</span></div></footer>
    </main>
  </div>;
}

export default App;
