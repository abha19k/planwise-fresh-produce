import { CommonModule } from '@angular/common';
import {
  AfterViewInit,
  Component,
  ElementRef,
  OnDestroy,
  OnInit,
  ViewChild,
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

type PeriodUI = 'Daily' | 'Weekly' | 'Monthly';
type Variant = 'baseline' | 'feat';

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
  x: string; // date string
  y: number;
}

@Component({
  selector: 'app-forecast-tuning',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    HttpClientModule,

    // CoreUI
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
  // ---------- API ----------
  apiBase = 'http://127.0.0.1:8000';

  // ---------- UI ----------
  loading = false;
  errorMessage = '';

  metrics: Record<string, number> | null = null;
  tuneResult: any | null = null;

  // ---------- Chart ----------
  @ViewChild('chartCanvas', { static: false })
  chartCanvas?: ElementRef<HTMLCanvasElement>;

  private chart: Chart | null = null;

  // ---------- Dropdown arrays (as used in HTML) ----------
  productIds: string[] = [];
  channelIds: string[] = [];
  locationIds: string[] = [];

  // ---------- Reactive controls (as used in HTML) ----------
  periodSelection = new FormControl<PeriodUI>('Daily', { nonNullable: true });

  productIdCtrl = new FormControl<string>('', { nonNullable: true });
  channelIdCtrl = new FormControl<string>('', { nonNullable: true });
  locationIdCtrl = new FormControl<string>('', { nonNullable: true });

  savedSearchId = new FormControl<number | null>(null);
  queryCtrl = new FormControl<string>({ value: '', disabled: true }, { nonNullable: true });
  newSavedName = new FormControl<string>('', { nonNullable: true });

  lagCtrl = new FormControl<number>(1, { nonNullable: true, validators: [Validators.min(1)] });
  horizonCtrl = new FormControl<number>(28, { nonNullable: true, validators: [Validators.min(1)] });
  foldsCtrl = new FormControl<number>(6, { nonNullable: true, validators: [Validators.min(1), Validators.max(6)] });
  useCleansedCtrl = new FormControl<boolean>(false, { nonNullable: true });

  // feature + params forms (as used in HTML)
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

  // ---------- Data series ----------
  histSeries: Point[] = [];
  fcSeries: Point[] = [];
  querySeries: Point[] = [];
  queryFcSeries: Point[] = [];

  // ---------- Saved searches ----------
  savedSearches: SavedSearch[] = [];

  constructor(private http: HttpClient) {}

  // =======================
  // lifecycle
  // =======================
  ngOnInit(): void {
    this.loading = true;

    forkJoin({
      products: this.http.get<any[]>(`${this.apiBase}/api/products`),
      channels: this.http.get<any[]>(`${this.apiBase}/api/channels`),
      locations: this.http.get<any[]>(`${this.apiBase}/api/locations`),
      saved: this.http.get<SavedSearch[]>(`${this.apiBase}/api/saved-searches`),
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

    // update query preview whenever any key changes
    this.periodSelection.valueChanges.subscribe(() => {
      this.clearPlotOnly();
      this.refreshQueryPreview();
      this.renderChart();
    });
    this.productIdCtrl.valueChanges.subscribe(() => this.refreshQueryPreview());
    this.channelIdCtrl.valueChanges.subscribe(() => this.refreshQueryPreview());
    this.locationIdCtrl.valueChanges.subscribe(() => this.refreshQueryPreview());

    // selecting saved search loads query string into queryCtrl
    this.savedSearchId.valueChanges.subscribe((id) => {
      const s = this.savedSearches.find((x) => x.id === id) ?? null;
      this.queryCtrl.setValue(s?.query ?? '', { emitEvent: false });
    });
  }

  ngAfterViewInit(): void {
    // create empty chart so "Load History" immediately updates it
    this.renderChart();
  }

  ngOnDestroy(): void {
    if (this.chart) {
      this.chart.destroy();
      this.chart = null;
    }
  }

  // =======================
  // misc helpers
  // =======================
  private fail(err: any, fallback: any) {
    console.error(err);
    this.errorMessage = String(err?.error?.detail ?? err?.message ?? 'Unknown error');
    this.loading = false;
    return of(fallback);
  }

  private getPeriod(): PeriodUI {
    return (this.periodSelection.value ?? 'Daily') as PeriodUI;
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
    // only set this auto query if user has NOT selected a saved search
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

  // ✅ NEW: clear only the "key" plot series
  private clearKeySeries(): void {
    this.histSeries = [];
    this.fcSeries = [];
  }

  // ✅ NEW: clear only the "query" plot series
  private clearQuerySeries(): void {
    this.querySeries = [];
    this.queryFcSeries = [];
  }

  // =======================
  // API wrappers
  // =======================
  private loadHistoryByKeys(keys: Key[], limitPerKey: number) {
    const period = this.getPeriod();
    const endpoint =
      period === 'Daily' ? 'daily-by-keys' : period === 'Weekly' ? 'weekly-by-keys' : 'monthly-by-keys';

    const url = `${this.apiBase}/api/history/${endpoint}`;
    const params = new HttpParams().set('limit_per_key', String(limitPerKey));

    return this.http
      .post<HistoryRow[]>(url, { keys }, { params })
      .pipe(map((rows) => (Array.isArray(rows) ? rows : [])));
  }

  private loadForecastForKey(key: Key, variant: Variant) {
    const params = new HttpParams()
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
    const params = new HttpParams().set('q', q).set('limit', '50000').set('offset', '0');
    return this.http.get<SearchKeysResponse>(`${this.apiBase}/api/search`, { params }).pipe(
      map((res) => (Array.isArray(res?.keys) ? res.keys : []))
    );
  }

  private loadHistoryByQuery(q: string) {
    const params = new HttpParams()
      .set('q', q)
      .set('limit', '5000')
      .set('offset', '0')
      .set('period', this.getPeriod());

    return this.http.get<HistoryByQueryResponse>(`${this.apiBase}/api/history/by-query`, { params }).pipe(
      map((res) => (Array.isArray(res?.rows) ? res.rows : []))
    );
  }

  // =======================
  // SERIES builders (null-safe)
  // =======================
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

  // =======================
  // CHART
  // =======================
  private renderChart(): void {
    const canvas = this.chartCanvas?.nativeElement;
    if (!canvas) return;

    const labels = this.unionLabels();

    const dsHist = this.projectToLabels(this.histSeries, labels);
    const dsFc = this.projectToLabels(this.fcSeries, labels);
    const dsHistQ = this.projectToLabels(this.querySeries, labels);
    const dsFcQ = this.projectToLabels(this.queryFcSeries, labels);

    // ✅ Only include datasets that actually have data (prevents "old" empty overlays)
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

  // =======================
  // TOP buttons (single key)
  // =======================
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

    // ✅ KEY mode should hide QUERY lines
    this.clearQuerySeries();

    this.loadHistoryByKeys([key], 2000)
      .pipe(
        tap((hist) => {
          this.histSeries = this.toHistorySeries(hist);
          this.fcSeries = []; // loading history only

          this.renderChart();
        }),
        finalize(() => (this.loading = false)),
        catchError((err) => this.fail(err, []))
      )
      .subscribe();
  }

  runSingleKeyForecast(save: boolean): void {
    // save flag currently unused unless you have an endpoint; plotting still works.
    this.errorMessage = '';
    this.metrics = null;
    this.tuneResult = null;

    const key = this.getSelectedKey();
    if (!key) {
      this.errorMessage = 'Select ProductID, ChannelID and LocationID.';
      return;
    }

    this.loading = true;

    // ✅ KEY mode should hide QUERY lines
    this.clearQuerySeries();

    this.loadHistoryByKeys([key], 2000)
      .pipe(
        tap((hist) => {
          this.histSeries = this.toHistorySeries(hist);
          this.renderChart();
        }),
        switchMap(() => this.loadForecastForKey(key, 'feat')),
        finalize(() => (this.loading = false)),
        catchError((err) => this.fail(err, []))
      )
      .subscribe((fc) => {
        this.fcSeries = this.toForecastSeries(fc);
        this.renderChart();
        if (save) {
          // If you implement a save endpoint later, call it here.
        }
      });
  }

  runBacktest(): void {
    this.errorMessage = 'Backtest not wired in this TS (needs backend endpoint).';
  }

  tuneXgb(): void {
    this.errorMessage = 'Tune (grid) not wired in this TS (needs backend endpoint).';
  }

  // =======================
  // SAVED SEARCH actions (bottom)
  // =======================
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
      .post<SavedSearch>(`${this.apiBase}/api/saved-searches`, { name, query })
      .pipe(
        switchMap(() => this.http.get<SavedSearch[]>(`${this.apiBase}/api/saved-searches`)),
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

    // ✅ QUERY mode should hide KEY lines
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

    // ✅ QUERY mode should hide KEY lines
    this.clearKeySeries();

    // We aggregate by keys matched by /api/search
    this.searchKeysByQuery(q)
      .pipe(
        switchMap((keys) => {
          const limited = keys.slice(0, 200);
          if (!limited.length) return of({ keys: limited, hist: [] as HistoryRow[] });

          return this.loadHistoryByKeys(limited, 800).pipe(
            map((hist) => ({ keys: limited, hist }))
          );
        }),
        tap(({ hist }) => {
          this.querySeries = this.toHistorySeries(hist);
          this.renderChart();
        }),
        switchMap(({ keys }) => {
          if (!keys.length) return of([] as ForecastRow[]);
          const calls = keys.slice(0, 200).map((k) => this.loadForecastForKey(k, 'feat'));
          return forkJoin(calls).pipe(map((lists) => lists.flat()));
        }),
        finalize(() => (this.loading = false)),
        catchError((err) => this.fail(err, []))
      )
      .subscribe((fcRows) => {
        this.queryFcSeries = this.toForecastSeries(fcRows);
        this.renderChart();
        if (save) {
          // hook save endpoint here later
        }
      });
  }

  // =======================
  // Defaults button
  // =======================
  setDefaultsForPeriod(p: PeriodUI): void {
    // your UI button calls this with periodSelection.value
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
