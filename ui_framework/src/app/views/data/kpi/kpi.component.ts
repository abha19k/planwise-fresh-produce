import { CommonModule } from '@angular/common';
import { HttpClient, HttpClientModule, HttpParams } from '@angular/common/http';
import { Component, OnDestroy, OnInit, inject } from '@angular/core';
import { FormControl, ReactiveFormsModule } from '@angular/forms';
import { catchError, finalize, forkJoin, map, of, switchMap } from 'rxjs';

import {
  ButtonDirective,
  CardBodyComponent,
  CardComponent,
  CardHeaderComponent,
  ColComponent,
  RowComponent,
  TableDirective,
} from '@coreui/angular';

import { ScenarioService } from '../../../services/scenario.service';

type PeriodUI = 'Daily' | 'Weekly' | 'Monthly';
type Variant = 'baseline' | 'feat';
type KpiTab = 'standard' | 'compare';

type KpiCard = {
  title: string;
  value: number;
  suffix?: string;
  hint?: string;
};

interface Key {
  ProductID: string;
  ChannelID: string;
  LocationID: string;
}

interface SavedSearch {
  id: number;
  name: string;
  query: string;
  created_at?: string;
}

interface ProductRow { ProductID: string; }
interface ChannelRow { ChannelID: string; }
interface LocationRow { LocationID: string; }

interface ScenarioRow {
  scenario_id: number;
  name: string;
  parent_scenario_id?: number | null;
  is_base?: boolean;
  created_at?: string | null;
  created_by?: string | null;
  status?: string;
}

interface HistoryRow {
  ProductID: string;
  ChannelID: string;
  LocationID: string;
  StartDate: string;
  EndDate: string;
  Period: string;
  Qty: number | string;
  Level?: string;
}

interface ForecastRow {
  ProductID: string;
  ChannelID: string;
  LocationID: string;
  StartDate: string;
  EndDate: string;
  Period: string;
  ForecastQty: number | string;
  Level?: string;
  Model?: string;
  Method?: string;
}

interface ForecastApiResponse {
  variant: string;
  count: number;
  rows: ForecastRow[];
}

interface SearchKeysResponse {
  query: string;
  count: number;
  keys: Key[];
}

interface Metrics {
  n: number;
  wape: number;
  mape: number;
  smape: number;
  mae: number;
  rmse: number;
  bias_pct: number;
  sae: number;
  saa: number;

  WAPE?: number;
  MAPE?: number;
  sMAPE?: number;
  MAE?: number;
  RMSE?: number;
  SAE?: number;
  SAA?: number;
  BIAS_PCT?: number;
}

interface KpiResult {
  title: string;
  period: PeriodUI;
  variant: Variant;
  lag: number;
  computedAt: string;
  scenarioId: number;
  scenarioName: string;

  key?: Key;
  query?: string;
  keys?: number;

  metrics: Metrics;
}

@Component({
  selector: 'app-kpi',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    HttpClientModule,
    RowComponent,
    ColComponent,
    CardComponent,
    CardHeaderComponent,
    CardBodyComponent,
    ButtonDirective,
    TableDirective,
  ],
  templateUrl: './kpi.component.html',
  styleUrls: ['./kpi.component.scss'],
})
export class KpiComponent implements OnInit, OnDestroy {
  private http = inject(HttpClient);
  readonly scenarioService = inject(ScenarioService);

  private readonly apiBase = 'http://127.0.0.1:8000';
  private readonly DB_SCHEMA = 'planwise_fresh_produce';

  loading = false;
  errorMessage = '';

  productIds: string[] = [];
  channelIds: string[] = [];
  locationIds: string[] = [];

  savedSearches: SavedSearch[] = [];
  scenarios: ScenarioRow[] = [];

  activeTab = new FormControl<KpiTab>('standard', { nonNullable: true });

  periodSelection = new FormControl<PeriodUI>('Daily', { nonNullable: true });
  variantSelection = new FormControl<Variant>('feat', { nonNullable: true });

  productIdCtrl = new FormControl<string>('', { nonNullable: true });
  channelIdCtrl = new FormControl<string>('', { nonNullable: true });
  locationIdCtrl = new FormControl<string>('', { nonNullable: true });

  savedSearchId = new FormControl<number | null>(null);
  queryCtrl = new FormControl<string>('', { nonNullable: true });

  lagCtrl = new FormControl<number>(1, { nonNullable: true });

  compareScenarioAId = new FormControl<number | null>(null);
  compareScenarioBId = new FormControl<number | null>(null);

  singleKpi: KpiResult | null = null;
  queryKpi: KpiResult | null = null;

  singleCards: KpiCard[] = [];
  queryCards: KpiCard[] = [];

  compareKeyKpiA: KpiResult | null = null;
  compareKeyKpiB: KpiResult | null = null;
  compareQueryKpiA: KpiResult | null = null;
  compareQueryKpiB: KpiResult | null = null;

  compareKeyCardsA: KpiCard[] = [];
  compareKeyCardsB: KpiCard[] = [];
  compareQueryCardsA: KpiCard[] = [];
  compareQueryCardsB: KpiCard[] = [];

  compareKeyDeltaCards: KpiCard[] = [];
  compareQueryDeltaCards: KpiCard[] = [];

  private subs: Array<{ unsubscribe: () => void }> = [];

  ngOnInit(): void {
    this.loading = true;

    forkJoin({
      products: this.http.get<ProductRow[]>(`${this.apiBase}/api/products`, {
        params: this.schemaParams()
      }),
      channels: this.http.get<ChannelRow[]>(`${this.apiBase}/api/channels`, {
        params: this.schemaParams()
      }),
      locations: this.http.get<LocationRow[]>(`${this.apiBase}/api/locations`, {
        params: this.schemaParams()
      }),
      saved: this.http.get<SavedSearch[]>(`${this.apiBase}/api/saved-searches`, {
        params: this.schemaParams()
      }),
      scenarios: this.http.get<ScenarioRow[]>(`${this.apiBase}/api/scenarios`, {
        params: this.schemaParams()
      }),
    })
      .pipe(
        map(({ products, channels, locations, saved, scenarios }) => {
          this.productIds = (products ?? []).map((r) => String(r.ProductID)).filter(Boolean);
          this.channelIds = (channels ?? []).map((r) => String(r.ChannelID)).filter(Boolean);
          this.locationIds = (locations ?? []).map((r) => String(r.LocationID)).filter(Boolean);
          this.savedSearches = Array.isArray(saved) ? saved : [];
          this.scenarios = Array.isArray(scenarios) ? scenarios : [];

          const currentSid = this.currentScenarioId;
          const a = this.scenarios.find(s => s.scenario_id === currentSid) ?? this.scenarios[0] ?? null;
          const b = this.scenarios.find(s => s.scenario_id !== (a?.scenario_id ?? -1)) ?? null;

          this.compareScenarioAId.setValue(a?.scenario_id ?? null, { emitEvent: false });
          this.compareScenarioBId.setValue(b?.scenario_id ?? null, { emitEvent: false });

          this.refreshQueryPreview();
          return true;
        }),
        finalize(() => (this.loading = false)),
        catchError((err) => this.fail(err, false))
      )
      .subscribe();

    this.subs.push(
      this.productIdCtrl.valueChanges.subscribe(() => this.refreshQueryPreview()),
      this.channelIdCtrl.valueChanges.subscribe(() => this.refreshQueryPreview()),
      this.locationIdCtrl.valueChanges.subscribe(() => this.refreshQueryPreview()),
      this.savedSearchId.valueChanges.subscribe((id) => {
        const s = this.savedSearches.find((x) => x.id === id) ?? null;
        this.queryCtrl.setValue(s?.query ?? '', { emitEvent: false });
      })
    );
  }

  ngOnDestroy(): void {
    for (const s of this.subs) s.unsubscribe();
  }

  get currentScenarioId(): number {
    return this.scenarioService.selectedScenarioId();
  }

  get currentScenarioName(): string {
    return this.scenarioService.selectedScenario()?.name || 'Base';
  }

  scenarioNameById(id: number | null): string {
    if (!id) return '—';
    return this.scenarios.find(s => s.scenario_id === id)?.name || `Scenario ${id}`;
  }

  clear(): void {
    this.errorMessage = '';

    this.singleKpi = null;
    this.queryKpi = null;
    this.singleCards = [];
    this.queryCards = [];

    this.compareKeyKpiA = null;
    this.compareKeyKpiB = null;
    this.compareQueryKpiA = null;
    this.compareQueryKpiB = null;

    this.compareKeyCardsA = [];
    this.compareKeyCardsB = [];
    this.compareQueryCardsA = [];
    this.compareQueryCardsB = [];

    this.compareKeyDeltaCards = [];
    this.compareQueryDeltaCards = [];
  }

  private fail(err: any, fallback: any) {
    console.error(err);
    this.errorMessage = String(err?.error?.detail ?? err?.message ?? 'Unknown error');
    this.loading = false;
    return of(fallback);
  }

  private schemaParams(): HttpParams {
    return new HttpParams()
      .set('db_schema', this.DB_SCHEMA);
  }

  private baseParamsForScenario(scenarioId: number): HttpParams {
    return new HttpParams()
      .set('db_schema', this.DB_SCHEMA)
      .set('scenario_id', scenarioId);
  }

  private getPeriod(): PeriodUI {
    return (this.periodSelection.value ?? 'Daily') as PeriodUI;
  }

  private getVariant(): Variant {
    return (this.variantSelection.value ?? 'feat') as Variant;
  }

  private getSelectedKey(): Key | null {
    const p = (this.productIdCtrl.value || '').trim();
    const c = (this.channelIdCtrl.value || '').trim();
    const l = (this.locationIdCtrl.value || '').trim();
    if (!p || !c || !l) return null;
    return { ProductID: p, ChannelID: c, LocationID: l };
  }

  private refreshQueryPreview(): void {
    if (this.savedSearchId.value) return;

    const key = this.getSelectedKey();
    if (!key) {
      this.queryCtrl.setValue('', { emitEvent: false });
      return;
    }

    const q = `productid:${key.ProductID} AND channelid:${key.ChannelID} AND locationid:${key.LocationID}`;
    this.queryCtrl.setValue(q, { emitEvent: false });
  }

  private toDateKey(dateStr: string): string {
    const s = String(dateStr || '');
    return s.length >= 10 ? s.slice(0, 10) : s;
  }

  private shiftDate(dateStr: string, period: PeriodUI, steps: number): string {
    const d = new Date(dateStr);
    if (Number.isNaN(d.getTime())) return this.toDateKey(dateStr);

    if (period === 'Daily') d.setDate(d.getDate() + steps);
    else if (period === 'Weekly') d.setDate(d.getDate() + 7 * steps);
    else d.setMonth(d.getMonth() + steps);

    const yyyy = d.getFullYear();
    const mm = String(d.getMonth() + 1).padStart(2, '0');
    const dd = String(d.getDate()).padStart(2, '0');
    return `${yyyy}-${mm}-${dd}`;
  }

  private emptyMetrics(): Metrics {
    return { n: 0, wape: 0, mape: 0, smape: 0, mae: 0, rmse: 0, bias_pct: 0, sae: 0, saa: 0 };
  }

  private withUpperAliases(m: Metrics): Metrics {
    return {
      ...m,
      WAPE: m.wape,
      MAPE: m.mape,
      sMAPE: m.smape,
      MAE: m.mae,
      RMSE: m.rmse,
      SAE: m.sae,
      SAA: m.saa,
      BIAS_PCT: m.bias_pct,
    };
  }

  private computeFromPairs(pairs: Array<{ a: number; f: number }>): Metrics {
    const n = pairs.length;
    if (!n) return this.emptyMetrics();

    let sae = 0;
    let saa = 0;
    let sse = 0;
    let sumMape = 0;
    let mapeN = 0;
    let sumSmape = 0;
    let smapeN = 0;
    let sumBias = 0;

    for (const { a, f } of pairs) {
      const e = f - a;
      const ae = Math.abs(e);

      sae += ae;
      saa += Math.abs(a);
      sse += e * e;
      sumBias += e;

      if (a !== 0) {
        sumMape += (ae / Math.abs(a)) * 100;
        mapeN += 1;
      }

      const denom = Math.abs(a) + Math.abs(f);
      if (denom !== 0) {
        sumSmape += (2 * ae / denom) * 100;
        smapeN += 1;
      }
    }

    const mae = sae / n;
    const rmse = Math.sqrt(sse / n);
    const wape = saa === 0 ? 0 : (sae / saa) * 100;
    const mape = mapeN ? (sumMape / mapeN) : 0;
    const smape = smapeN ? (sumSmape / smapeN) : 0;
    const bias_pct = saa === 0 ? 0 : (sumBias / saa) * 100;

    return { n, sae, saa, mae, rmse, wape, mape, smape, bias_pct };
  }

  private computeSingleMetricsWithLag(hist: HistoryRow[], fc: ForecastRow[], lag: number): Metrics {
    const period = this.getPeriod();

    const actualByDate = new Map<string, number>();
    for (const r of hist ?? []) {
      actualByDate.set(this.toDateKey(r.StartDate), Number(r.Qty ?? 0) || 0);
    }

    const pairs: Array<{ a: number; f: number }> = [];
    for (const r of fc ?? []) {
      const fcDate = this.toDateKey(r.StartDate);
      const histDate = this.shiftDate(fcDate, period, -lag);
      if (!actualByDate.has(histDate)) continue;

      pairs.push({
        a: actualByDate.get(histDate) ?? 0,
        f: Number(r.ForecastQty ?? 0) || 0,
      });
    }

    return this.computeFromPairs(pairs);
  }

  private computeQueryMetricsWithLag(hist: HistoryRow[], fc: ForecastRow[], lag: number): Metrics {
    const period = this.getPeriod();

    const actualByKeyDate = new Map<string, number>();
    for (const r of hist ?? []) {
      const k = `${r.ProductID}|${r.ChannelID}|${r.LocationID}|${this.toDateKey(r.StartDate)}`;
      actualByKeyDate.set(k, Number(r.Qty ?? 0) || 0);
    }

    const pairs: Array<{ a: number; f: number }> = [];
    for (const r of fc ?? []) {
      const fcDate = this.toDateKey(r.StartDate);
      const histDate = this.shiftDate(fcDate, period, -lag);
      const k = `${r.ProductID}|${r.ChannelID}|${r.LocationID}|${histDate}`;
      if (!actualByKeyDate.has(k)) continue;

      pairs.push({
        a: actualByKeyDate.get(k) ?? 0,
        f: Number(r.ForecastQty ?? 0) || 0,
      });
    }

    return this.computeFromPairs(pairs);
  }

  private buildCards(m: Metrics): KpiCard[] {
    return [
      { title: 'n', value: m.n, hint: 'Matched points after lag alignment' },
      { title: 'WAPE', value: m.wape, suffix: '%', hint: 'Sum|err| / Sum|actual| × 100' },
      { title: 'MAPE', value: m.mape, suffix: '%', hint: 'Mean(|err|/|actual|) × 100 (actual≠0)' },
      { title: 'sMAPE', value: m.smape, suffix: '%', hint: 'Mean(2|err|/(|a|+|f|)) × 100' },
      { title: 'MAE', value: m.mae, hint: 'Mean absolute error' },
      { title: 'RMSE', value: m.rmse, hint: 'Root mean squared error' },
      { title: 'Bias', value: m.bias_pct, suffix: '%', hint: 'Sum(err)/Sum|actual| × 100' },
    ];
  }

  private buildDeltaCards(a: Metrics, b: Metrics): KpiCard[] {
    return [
      { title: 'Δ WAPE', value: b.wape - a.wape, suffix: '%', hint: 'Scenario B - Scenario A' },
      { title: 'Δ MAPE', value: b.mape - a.mape, suffix: '%', hint: 'Scenario B - Scenario A' },
      { title: 'Δ sMAPE', value: b.smape - a.smape, suffix: '%', hint: 'Scenario B - Scenario A' },
      { title: 'Δ MAE', value: b.mae - a.mae, hint: 'Scenario B - Scenario A' },
      { title: 'Δ RMSE', value: b.rmse - a.rmse, hint: 'Scenario B - Scenario A' },
      { title: 'Δ Bias', value: b.bias_pct - a.bias_pct, suffix: '%', hint: 'Scenario B - Scenario A' },
    ];
  }

  private loadHistoryByKeys(keys: Key[], limitPerKey: number) {
    const period = this.getPeriod();
    const endpoint =
      period === 'Daily' ? 'daily-by-keys' : period === 'Weekly' ? 'weekly-by-keys' : 'monthly-by-keys';

    const url = `${this.apiBase}/api/history/${endpoint}`;
    const params = this.schemaParams().set('limit_per_key', String(limitPerKey));

    return this.http
      .post<HistoryRow[]>(url, { keys }, { params })
      .pipe(map((rows) => (Array.isArray(rows) ? rows : [])));
  }

  private loadForecastForKey(key: Key, variant: Variant, scenarioId: number) {
    const params = this.baseParamsForScenario(scenarioId)
      .set('variant', variant)
      .set('productid', key.ProductID)
      .set('channelid', key.ChannelID)
      .set('locationid', key.LocationID)
      .set('period', this.getPeriod())
      .set('limit', '5000')
      .set('offset', '0');

    return this.http.get<ForecastApiResponse>(`${this.apiBase}/api/forecast`, { params }).pipe(
      map((res) => (Array.isArray(res?.rows) ? res.rows : []))
    );
  }

  private searchKeysByQuery(q: string) {
    const params = this.schemaParams()
      .set('q', q)
      .set('limit', '50000')
      .set('offset', '0');

    return this.http.get<SearchKeysResponse>(`${this.apiBase}/api/search`, { params }).pipe(
      map((res) => (Array.isArray(res?.keys) ? res.keys : []))
    );
  }

  computeSingleKpi(): void {
    this.errorMessage = '';
    this.singleKpi = null;
    this.singleCards = [];

    const key = this.getSelectedKey();
    if (!key) {
      this.errorMessage = 'Select ProductID, ChannelID and LocationID.';
      return;
    }

    const lag = Math.max(0, Number(this.lagCtrl.value ?? 0));
    const period = this.getPeriod();
    const variant = this.getVariant();
    const scenarioId = this.currentScenarioId;

    this.loading = true;

    this.loadHistoryByKeys([key], 5000)
      .pipe(
        switchMap((hist) =>
          this.loadForecastForKey(key, variant, scenarioId).pipe(
            map((fc) => ({ hist, fc }))
          )
        ),
        finalize(() => (this.loading = false)),
        catchError((err) => this.fail(err, { hist: [] as HistoryRow[], fc: [] as ForecastRow[] }))
      )
      .subscribe(({ hist, fc }) => {
        const metrics = this.withUpperAliases(this.computeSingleMetricsWithLag(hist, fc, lag));

        if (metrics.n === 0) {
          this.errorMessage = `No overlapping actual/forecast pairs found for scenario ${scenarioId}.`;
        }

        this.singleKpi = {
          title: 'Single KPI',
          period,
          variant,
          lag,
          computedAt: new Date().toISOString(),
          scenarioId,
          scenarioName: this.currentScenarioName,
          key,
          metrics,
        };

        this.singleCards = this.buildCards(metrics);
      });
  }

  computeQueryKpi(): void {
    this.errorMessage = '';
    this.queryKpi = null;
    this.queryCards = [];

    const q = (this.queryCtrl.value || '').trim();
    if (!q) {
      this.errorMessage = 'Choose a saved search or build a query first.';
      return;
    }

    const lag = Math.max(0, Number(this.lagCtrl.value ?? 0));
    const period = this.getPeriod();
    const variant = this.getVariant();
    const scenarioId = this.currentScenarioId;

    this.loading = true;

    this.searchKeysByQuery(q)
      .pipe(
        switchMap((keys) => {
          const limited = keys.slice(0, 200);
          if (!limited.length) {
            return of({ keys: limited, hist: [] as HistoryRow[], fc: [] as ForecastRow[] });
          }

          return this.loadHistoryByKeys(limited, 2000).pipe(
            switchMap((hist) => {
              const calls = limited.map((k) => this.loadForecastForKey(k, variant, scenarioId));
              return forkJoin(calls).pipe(
                map((lists) => lists.flat()),
                map((fc) => ({ keys: limited, hist, fc }))
              );
            })
          );
        }),
        finalize(() => (this.loading = false)),
        catchError((err) => this.fail(err, { keys: [] as Key[], hist: [] as HistoryRow[], fc: [] as ForecastRow[] }))
      )
      .subscribe(({ keys, hist, fc }) => {
        const metrics = this.withUpperAliases(this.computeQueryMetricsWithLag(hist, fc, lag));

        if (metrics.n === 0) {
          this.errorMessage = `No overlapping actual/forecast pairs found for scenario ${scenarioId}.`;
        }

        this.queryKpi = {
          title: 'Query KPI',
          period,
          variant,
          lag,
          computedAt: new Date().toISOString(),
          scenarioId,
          scenarioName: this.currentScenarioName,
          query: q,
          keys: keys.length,
          metrics,
        };

        this.queryCards = this.buildCards(metrics);
      });
  }

  computeCompareKeyKpi(): void {
    this.errorMessage = '';
    this.compareKeyKpiA = null;
    this.compareKeyKpiB = null;
    this.compareKeyCardsA = [];
    this.compareKeyCardsB = [];
    this.compareKeyDeltaCards = [];

    const key = this.getSelectedKey();
    if (!key) {
      this.errorMessage = 'Select ProductID, ChannelID and LocationID.';
      return;
    }

    const scenarioA = this.compareScenarioAId.value;
    const scenarioB = this.compareScenarioBId.value;

    if (!scenarioA || !scenarioB) {
      this.errorMessage = 'Choose both Scenario A and Scenario B.';
      return;
    }

    const lag = Math.max(0, Number(this.lagCtrl.value ?? 0));
    const period = this.getPeriod();
    const variant = this.getVariant();

    this.loading = true;

    this.loadHistoryByKeys([key], 5000)
      .pipe(
        switchMap((hist) =>
          forkJoin({
            fcA: this.loadForecastForKey(key, variant, scenarioA),
            fcB: this.loadForecastForKey(key, variant, scenarioB),
          }).pipe(map(({ fcA, fcB }) => ({ hist, fcA, fcB })))
        ),
        finalize(() => (this.loading = false)),
        catchError((err) => this.fail(err, null))
      )
      .subscribe((payload) => {
        if (!payload) return;

        const metricsA = this.withUpperAliases(this.computeSingleMetricsWithLag(payload.hist, payload.fcA, lag));
        const metricsB = this.withUpperAliases(this.computeSingleMetricsWithLag(payload.hist, payload.fcB, lag));

        this.compareKeyKpiA = {
          title: 'Scenario A KPI',
          period,
          variant,
          lag,
          computedAt: new Date().toISOString(),
          scenarioId: scenarioA,
          scenarioName: this.scenarioNameById(scenarioA),
          key,
          metrics: metricsA,
        };

        this.compareKeyKpiB = {
          title: 'Scenario B KPI',
          period,
          variant,
          lag,
          computedAt: new Date().toISOString(),
          scenarioId: scenarioB,
          scenarioName: this.scenarioNameById(scenarioB),
          key,
          metrics: metricsB,
        };

        this.compareKeyCardsA = this.buildCards(metricsA);
        this.compareKeyCardsB = this.buildCards(metricsB);
        this.compareKeyDeltaCards = this.buildDeltaCards(metricsA, metricsB);

        if (metricsA.n === 0 && metricsB.n === 0) {
          this.errorMessage = 'No overlapping actual/forecast pairs found for either scenario.';
        }
      });
  }

  computeCompareQueryKpi(): void {
    this.errorMessage = '';
    this.compareQueryKpiA = null;
    this.compareQueryKpiB = null;
    this.compareQueryCardsA = [];
    this.compareQueryCardsB = [];
    this.compareQueryDeltaCards = [];

    const q = (this.queryCtrl.value || '').trim();
    if (!q) {
      this.errorMessage = 'Choose a saved search or build a query first.';
      return;
    }

    const scenarioA = this.compareScenarioAId.value;
    const scenarioB = this.compareScenarioBId.value;

    if (!scenarioA || !scenarioB) {
      this.errorMessage = 'Choose both Scenario A and Scenario B.';
      return;
    }

    const lag = Math.max(0, Number(this.lagCtrl.value ?? 0));
    const period = this.getPeriod();
    const variant = this.getVariant();

    this.loading = true;

    this.searchKeysByQuery(q)
      .pipe(
        switchMap((keys) => {
          const limited = keys.slice(0, 200);
          if (!limited.length) {
            return of({
              keys: limited,
              hist: [] as HistoryRow[],
              fcA: [] as ForecastRow[],
              fcB: [] as ForecastRow[],
            });
          }

          return this.loadHistoryByKeys(limited, 2000).pipe(
            switchMap((hist) => {
              const callsA = limited.map((k) => this.loadForecastForKey(k, variant, scenarioA));
              const callsB = limited.map((k) => this.loadForecastForKey(k, variant, scenarioB));

              return forkJoin({
                fcA: forkJoin(callsA).pipe(map((lists) => lists.flat())),
                fcB: forkJoin(callsB).pipe(map((lists) => lists.flat())),
              }).pipe(map(({ fcA, fcB }) => ({
                keys: limited,
                hist,
                fcA,
                fcB,
              })));
            })
          );
        }),
        finalize(() => (this.loading = false)),
        catchError((err) => this.fail(err, {
          keys: [] as Key[],
          hist: [] as HistoryRow[],
          fcA: [] as ForecastRow[],
          fcB: [] as ForecastRow[],
        }))
      )
      .subscribe(({ keys, hist, fcA, fcB }) => {
        const metricsA = this.withUpperAliases(this.computeQueryMetricsWithLag(hist, fcA, lag));
        const metricsB = this.withUpperAliases(this.computeQueryMetricsWithLag(hist, fcB, lag));

        this.compareQueryKpiA = {
          title: 'Scenario A KPI',
          period,
          variant,
          lag,
          computedAt: new Date().toISOString(),
          scenarioId: scenarioA,
          scenarioName: this.scenarioNameById(scenarioA),
          query: q,
          keys: keys.length,
          metrics: metricsA,
        };

        this.compareQueryKpiB = {
          title: 'Scenario B KPI',
          period,
          variant,
          lag,
          computedAt: new Date().toISOString(),
          scenarioId: scenarioB,
          scenarioName: this.scenarioNameById(scenarioB),
          query: q,
          keys: keys.length,
          metrics: metricsB,
        };

        this.compareQueryCardsA = this.buildCards(metricsA);
        this.compareQueryCardsB = this.buildCards(metricsB);
        this.compareQueryDeltaCards = this.buildDeltaCards(metricsA, metricsB);

        if (metricsA.n === 0 && metricsB.n === 0) {
          this.errorMessage = 'No overlapping actual/forecast pairs found for either scenario.';
        }
      });
  }
}