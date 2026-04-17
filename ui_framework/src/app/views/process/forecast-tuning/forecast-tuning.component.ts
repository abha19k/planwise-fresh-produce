import { CommonModule } from '@angular/common';
import {
  AfterViewInit,
  Component,
  ElementRef,
  OnDestroy,
  OnInit,
  ViewChild,
  inject,
} from '@angular/core';
import {
  FormControl,
  FormGroup,
  ReactiveFormsModule,
  Validators,
} from '@angular/forms';
import { HttpClient, HttpClientModule, HttpParams } from '@angular/common/http';

import { catchError, finalize, forkJoin, map, of, switchMap, tap } from 'rxjs';

import Chart from 'chart.js/auto';

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
type LevelUI = '111' | '121' | '221';

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

interface HistoryRow {
  ProductID: string;
  ChannelID: string;
  LocationID: string;
  StartDate: string;
  EndDate: string;
  Period: string;
  Qty: number | string;
  Level?: string;
  NetPrice?: number | string | null;
  ListPrice?: number | string | null;
  Type?: string;
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
  NetPrice?: number | string | null;
  ListPrice?: number | string | null;
  UOM?: string;
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

interface HistoryByQueryResponse {
  q: string;
  count: number;
  rows: HistoryRow[];
}

interface Point {
  x: string;
  y: number;
}

interface RunOneDbResponse {
  ok: boolean;
  scenario_id: number;
  message: string;
  tag: string;
  db_result?: any;
  rows_baseline?: number;
  rows_feat?: number;
  backtest_rows?: number;
  mean_wmape_base?: number;
  mean_wmape_feat?: number;
}

@Component({
  selector: 'app-forecast-tuning',
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
  templateUrl: './forecast-tuning.component.html',
  styleUrls: ['./forecast-tuning.component.scss'],
})
export class ForecastTuningComponent implements OnInit, AfterViewInit, OnDestroy {
  private http = inject(HttpClient);

  readonly scenarioService = inject(ScenarioService);

  private readonly apiBase = 'http://127.0.0.1:8000';
  private readonly DB_SCHEMA = 'planwise_fresh_produce';

  loading = false;
  errorMessage = '';

  metrics: Record<string, number> | null = null;
  tuneResult: any | null = null;

  @ViewChild('chartCanvas', { static: false })
  chartCanvas?: ElementRef<HTMLCanvasElement>;

  private chart: Chart | null = null;

  productIds: string[] = [];
  channelIds: string[] = [];
  locationIds: string[] = [];
  levels: LevelUI[] = ['111', '121', '221'];

  periodSelection = new FormControl<PeriodUI>('Daily', { nonNullable: true });
  levelSelection = new FormControl<LevelUI>('111', { nonNullable: true });

  productIdCtrl = new FormControl<string>('', { nonNullable: true });
  channelIdCtrl = new FormControl<string>('', { nonNullable: true });
  locationIdCtrl = new FormControl<string>('', { nonNullable: true });

  savedSearchId = new FormControl<number | null>(null);
  queryCtrl = new FormControl<string>({ value: '', disabled: true }, { nonNullable: true });
  newSavedName = new FormControl<string>('', { nonNullable: true });

  lagCtrl = new FormControl<number>(1, {
    nonNullable: true,
    validators: [Validators.min(1)]
  });

  horizonCtrl = new FormControl<number>(28, {
    nonNullable: true,
    validators: [Validators.min(1)]
  });

  foldsCtrl = new FormControl<number>(6, {
    nonNullable: true,
    validators: [Validators.min(1), Validators.max(6)]
  });

  useCleansedCtrl = new FormControl<boolean>(false, { nonNullable: true });

  xgbFeaturesForm = new FormGroup({
    seasonal_lags: new FormControl<string>(''),
    rolling_windows: new FormControl<string>(''),
    use_log1p: new FormControl<boolean>(false, { nonNullable: true }),
    two_stage: new FormControl<boolean>(false, { nonNullable: true }),
    zero_prob_threshold: new FormControl<number>(0.5, { nonNullable: true }),
  });

  xgbParamsForm = new FormGroup({
    n_estimators: new FormControl<number>(300, { nonNullable: true }),
    learning_rate: new FormControl<number>(0.05, { nonNullable: true }),
    max_depth: new FormControl<number>(6, { nonNullable: true }),
    subsample: new FormControl<number>(0.9, { nonNullable: true }),
    colsample_bytree: new FormControl<number>(0.9, { nonNullable: true }),
    reg_lambda: new FormControl<number>(1.0, { nonNullable: true }),
    reg_alpha: new FormControl<number>(0.0, { nonNullable: true }),
    min_child_weight: new FormControl<number>(1.0, { nonNullable: true }),
    gamma: new FormControl<number>(0.0, { nonNullable: true }),
  });

  histSeries: Point[] = [];
  fcSeries: Point[] = [];
  querySeries: Point[] = [];
  queryFcSeries: Point[] = [];

  savedSearches: SavedSearch[] = [];

  ngOnInit(): void {
    this.loading = true;

    forkJoin({
      products: this.http.get<any[]>(`${this.apiBase}/api/products`, {
        params: this.schemaParams()
      }),
      channels: this.http.get<any[]>(`${this.apiBase}/api/channels`, {
        params: this.schemaParams()
      }),
      locations: this.http.get<any[]>(`${this.apiBase}/api/locations`, {
        params: this.schemaParams()
      }),
      saved: this.http.get<SavedSearch[]>(`${this.apiBase}/api/saved-searches`, {
        params: this.schemaParams()
      }),
    })
      .pipe(
        tap(({ products, channels, locations, saved }) => {
          this.productIds = (products ?? []).map((r) => String(r.ProductID)).filter(Boolean);
          this.channelIds = (channels ?? []).map((r) => String(r.ChannelID)).filter(Boolean);
          this.locationIds = (locations ?? []).map((r) => String(r.LocationID)).filter(Boolean);
          this.savedSearches = Array.isArray(saved) ? saved : [];
          this.refreshQueryPreview();
        }),
        finalize(() => (this.loading = false)),
        catchError((err) => this.fail(err, null))
      )
      .subscribe();

    this.periodSelection.valueChanges.subscribe(() => {
      this.clearPlotOnly();
      this.refreshQueryPreview();
      this.renderChart();
    });

    this.levelSelection.valueChanges.subscribe(() => {
      this.clearPlotOnly();
      this.renderChart();
    });

    this.productIdCtrl.valueChanges.subscribe(() => this.refreshQueryPreview());
    this.channelIdCtrl.valueChanges.subscribe(() => this.refreshQueryPreview());
    this.locationIdCtrl.valueChanges.subscribe(() => this.refreshQueryPreview());

    this.savedSearchId.valueChanges.subscribe((id) => {
      const s = this.savedSearches.find((x) => x.id === id) ?? null;
      this.queryCtrl.setValue(s?.query ?? '', { emitEvent: false });
    });
  }

  ngAfterViewInit(): void {
    this.renderChart();
  }

  ngOnDestroy(): void {
    if (this.chart) {
      this.chart.destroy();
      this.chart = null;
    }
  }

  get currentScenarioId(): number {
    return this.scenarioService.selectedScenarioId();
  }

  private schemaParams(): HttpParams {
    return new HttpParams().set('db_schema', this.DB_SCHEMA);
  }

  private baseParams(): HttpParams {
    return new HttpParams()
      .set('db_schema', this.DB_SCHEMA)
      .set('scenario_id', this.currentScenarioId);
  }

  private fail(err: any, fallback: any) {
    console.error(err);
    this.errorMessage = String(err?.error?.detail ?? err?.message ?? 'Unknown error');
    this.loading = false;
    return of(fallback);
  }

  private getPeriod(): PeriodUI {
    return (this.periodSelection.value ?? 'Daily') as PeriodUI;
  }

  private getLevel(): LevelUI {
    return (this.levelSelection.value ?? '111') as LevelUI;
  }

  private getSelectedKey(): Key | null {
    const p = (this.productIdCtrl.value || '').trim();
    const c = (this.channelIdCtrl.value || '').trim();
    const l = (this.locationIdCtrl.value || '').trim();
    if (!p || !c || !l) return null;
    return { ProductID: p, ChannelID: c, LocationID: l };
  }

  private refreshQueryPreview(): void {
    const key = this.getSelectedKey();
    if (!key) {
      this.queryCtrl.setValue('', { emitEvent: false });
      return;
    }

    const q = `productid:${key.ProductID} AND channelid:${key.ChannelID} AND locationid:${key.LocationID}`;

    if (!this.savedSearchId.value) {
      this.queryCtrl.setValue(q, { emitEvent: false });
    }
  }

  private clearPlotOnly(): void {
    this.histSeries = [];
    this.fcSeries = [];
    this.querySeries = [];
    this.queryFcSeries = [];
  }

  private clearKeySeries(): void {
    this.histSeries = [];
    this.fcSeries = [];
  }

  private clearQuerySeries(): void {
    this.querySeries = [];
    this.queryFcSeries = [];
  }

  private buildRunOneDbBody(saveToDb: boolean): any {
    const level = this.getLevel();
    const period = this.getPeriod();
    const horizon = Math.max(1, Number(this.horizonCtrl.value ?? 1));

    return {
      db_schema: this.DB_SCHEMA,
      scenario_id: this.currentScenarioId,
      level,
      period,
      horizon,
      weather_table: 'weather_daily',
      promo_table: 'promotions',
      tag: `${level}_${period}_scenario_${this.currentScenarioId}`,
      save_to_db: saveToDb
    };
  }

  private runForecastJob(saveToDb: boolean) {
    const body = this.buildRunOneDbBody(saveToDb);

    return this.http.post<RunOneDbResponse>(
      `${this.apiBase}/forecast/run-one-db`,
      body
    );
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

  private loadForecastForKey(key: Key, variant: Variant) {
    const params = this.baseParams()
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

  private loadHistoryByQuery(q: string) {
    const params = this.schemaParams()
      .set('q', q)
      .set('limit', '5000')
      .set('offset', '0')
      .set('period', this.getPeriod());

    return this.http.get<HistoryByQueryResponse>(`${this.apiBase}/api/history/by-query`, { params }).pipe(
      map((res) => (Array.isArray(res?.rows) ? res.rows : []))
    );
  }

  private toHistorySeries(rows: HistoryRow[] | null | undefined): Point[] {
    const arr = Array.isArray(rows) ? rows : [];
    return [...arr]
      .sort((a, b) => String(a.StartDate).localeCompare(String(b.StartDate)))
      .map((r) => ({ x: String(r.StartDate), y: Number(r.Qty ?? 0) || 0 }));
  }

  private toForecastSeries(rows: ForecastRow[] | null | undefined): Point[] {
    const arr = Array.isArray(rows) ? rows : [];
    return [...arr]
      .sort((a, b) => String(a.StartDate).localeCompare(String(b.StartDate)))
      .map((r) => ({ x: String(r.StartDate), y: Number(r.ForecastQty ?? 0) || 0 }));
  }

  private renderChart(): void {
    const canvas = this.chartCanvas?.nativeElement;
    if (!canvas) return;

    const labels = this.unionLabels();

    const dsHist = this.projectToLabels(this.histSeries, labels);
    const dsFc = this.projectToLabels(this.fcSeries, labels);
    const dsHistQ = this.projectToLabels(this.querySeries, labels);
    const dsFcQ = this.projectToLabels(this.queryFcSeries, labels);

    const datasets: any[] = [];
    if (this.histSeries.length) datasets.push({ label: 'History', data: dsHist });
    if (this.fcSeries.length) datasets.push({ label: 'Forecast', data: dsFc });
    if (this.querySeries.length) datasets.push({ label: 'History (Query)', data: dsHistQ });
    if (this.queryFcSeries.length) datasets.push({ label: 'Forecast (Query)', data: dsFcQ });

    const data = { labels, datasets };

    if (this.chart) {
      this.chart.data = data as any;
      this.chart.update();
      return;
    }

    this.chart = new Chart(canvas, {
      type: 'line',
      data: data as any,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: true } },
        scales: { y: { beginAtZero: true } },
      },
    });
  }

  private unionLabels(): string[] {
    const s = new Set<string>();
    for (const p of this.histSeries) s.add(p.x);
    for (const p of this.fcSeries) s.add(p.x);
    for (const p of this.querySeries) s.add(p.x);
    for (const p of this.queryFcSeries) s.add(p.x);
    return Array.from(s).sort((a, b) => a.localeCompare(b));
  }

  private projectToLabels(series: Point[], labels: string[]): Array<number | null> {
    const m = new Map(series.map((p) => [p.x, p.y]));
    return labels.map((x) => (m.has(x) ? (m.get(x) as number) : null));
  }

  updatePlot(): void {
    this.errorMessage = '';
    this.metrics = null;
    this.tuneResult = null;

    const key = this.getSelectedKey();
    if (!key) {
      this.errorMessage = 'Select ProductID, ChannelID and LocationID.';
      return;
    }

    this.loading = true;
    this.clearQuerySeries();

    this.loadHistoryByKeys([key], 2000)
      .pipe(
        tap((hist) => {
          this.histSeries = this.toHistorySeries(hist);
          this.fcSeries = [];
          this.renderChart();
        }),
        finalize(() => (this.loading = false)),
        catchError((err) => this.fail(err, []))
      )
      .subscribe();
  }

  runSingleKeyForecast(save: boolean): void {
    this.errorMessage = '';
    this.metrics = null;
    this.tuneResult = null;

    const key = this.getSelectedKey();
    if (!key) {
      this.errorMessage = 'Select ProductID, ChannelID and LocationID.';
      return;
    }

    this.loading = true;
    this.clearQuerySeries();

    this.loadHistoryByKeys([key], 2000)
      .pipe(
        tap((hist) => {
          this.histSeries = this.toHistorySeries(hist);
          this.renderChart();
        }),
        switchMap(() => this.runForecastJob(save)),
        tap((runRes) => {
          this.tuneResult = runRes;
        }),
        switchMap((runRes) => {
          if (!runRes?.ok) {
            throw new Error(runRes?.message || 'Forecast run failed');
          }

          return this.loadForecastForKey(key, 'feat');
        }),
        finalize(() => (this.loading = false)),
        catchError((err) => this.fail(err, []))
      )
      .subscribe((fc) => {
        this.fcSeries = this.toForecastSeries(fc);
        this.renderChart();
      });
  }

  runBacktest(): void {
    this.errorMessage = 'Backtest not wired in this TS (needs backend endpoint).';
  }

  tuneXgb(): void {
    this.errorMessage = 'Tune (grid) not wired in this TS (needs backend endpoint).';
  }

  saveCurrentAsSearch(): void {
    this.errorMessage = '';

    const name = (this.newSavedName.value || '').trim();
    const query = (this.queryCtrl.value || '').trim();

    if (!name) {
      this.errorMessage = 'Enter a name to save the search.';
      return;
    }
    if (!query) {
      this.errorMessage = 'No query to save.';
      return;
    }

    this.loading = true;

    this.http
      .post<SavedSearch>(`${this.apiBase}/api/saved-searches`, { name, query }, {
        params: this.schemaParams()
      })
      .pipe(
        switchMap(() =>
          this.http.get<SavedSearch[]>(`${this.apiBase}/api/saved-searches`, {
            params: this.schemaParams()
          })
        ),
        finalize(() => (this.loading = false)),
        catchError((err) => this.fail(err, []))
      )
      .subscribe((rows) => {
        this.savedSearches = Array.isArray(rows) ? rows : [];
        this.newSavedName.setValue('');
      });
  }

  loadHistoryForQuery(): void {
    this.errorMessage = '';
    this.metrics = null;
    this.tuneResult = null;

    const q = (this.queryCtrl.value || '').trim();
    if (!q) {
      this.errorMessage = 'Choose a saved search or select Product/Channel/Location first.';
      return;
    }

    this.loading = true;
    this.clearKeySeries();

    this.loadHistoryByQuery(q)
      .pipe(
        finalize(() => (this.loading = false)),
        catchError((err) => this.fail(err, []))
      )
      .subscribe((rows) => {
        this.querySeries = this.toHistorySeries(rows);
        this.queryFcSeries = [];
        this.renderChart();
      });
  }

  runForecastForQuery(save: boolean): void {
    this.errorMessage = '';
    this.metrics = null;
    this.tuneResult = null;

    const q = (this.queryCtrl.value || '').trim();
    if (!q) {
      this.errorMessage = 'Choose a saved search or build a query first.';
      return;
    }

    this.loading = true;
    this.clearKeySeries();

    this.searchKeysByQuery(q)
      .pipe(
        switchMap((keys) => {
          const limited = keys.slice(0, 200);
          if (!limited.length) {
            return of({ keys: limited, hist: [] as HistoryRow[] });
          }

          return this.loadHistoryByKeys(limited, 800).pipe(
            map((hist) => ({ keys: limited, hist }))
          );
        }),
        tap(({ hist }) => {
          this.querySeries = this.toHistorySeries(hist);
          this.renderChart();
        }),
        switchMap(({ keys }) => {
          if (!keys.length) {
            return of({ keys, fcRows: [] as ForecastRow[] });
          }

          return this.runForecastJob(save).pipe(
            tap((runRes) => {
              this.tuneResult = runRes;
            }),
            switchMap((runRes) => {
              if (!runRes?.ok) {
                throw new Error(runRes?.message || 'Forecast run failed');
              }

              const calls = keys.slice(0, 200).map((k) => this.loadForecastForKey(k, 'feat'));
              return forkJoin(calls).pipe(
                map((lists) => ({ keys, fcRows: lists.flat() }))
              );
            })
          );
        }),
        finalize(() => (this.loading = false)),
        catchError((err) => this.fail(err, { keys: [] as Key[], fcRows: [] as ForecastRow[] }))
      )
      .subscribe(({ fcRows }) => {
        this.queryFcSeries = this.toForecastSeries(fcRows);
        this.renderChart();
      });
  }

  setDefaultsForPeriod(p: PeriodUI): void {
    if (p === 'Daily') {
      this.lagCtrl.setValue(1);
      this.horizonCtrl.setValue(28);
      this.foldsCtrl.setValue(6);
    } else if (p === 'Weekly') {
      this.lagCtrl.setValue(1);
      this.horizonCtrl.setValue(13);
      this.foldsCtrl.setValue(6);
    } else {
      this.lagCtrl.setValue(1);
      this.horizonCtrl.setValue(12);
      this.foldsCtrl.setValue(3);
    }
  }
}