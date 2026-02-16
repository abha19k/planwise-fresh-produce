// // src/app/views/data/kpi/kpi.component.ts
// // ✅ Makes TS MATCH the HTML errors you posted:
// //   - adds variantSelection
// //   - adds clear()
// //   - adds singleCards / queryCards arrays
// //   - adds singleKpi.key, queryKpi.query, queryKpi.keys
// //   - supports BOTH metrics.wape and metrics.WAPE (uppercase aliases)
// // ✅ computes KPIs using LAG alignment so n != 0 when shifted matches exist

// import { CommonModule } from '@angular/common';
// import { HttpClient, HttpClientModule, HttpParams } from '@angular/common/http';
// import { Component, OnDestroy, OnInit } from '@angular/core';
// import { FormControl, ReactiveFormsModule } from '@angular/forms';
// import { catchError, finalize, forkJoin, map, of, switchMap } from 'rxjs';

// // CoreUI standalone imports
// import {
//   ButtonDirective,
//   CardBodyComponent,
//   CardComponent,
//   CardHeaderComponent,
//   ColComponent,
//   RowComponent,
//   TableDirective,
// } from '@coreui/angular';

// type PeriodUI = 'Daily' | 'Weekly' | 'Monthly';
// type Variant = 'baseline' | 'feat';
// type KpiCard = {
//     title: string;
//     value: number;
//     suffix?: string;
//     hint?: string;
//   };

// interface Key {
//   ProductID: string;
//   ChannelID: string;
//   LocationID: string;
// }

// interface SavedSearch {
//   id: number;
//   name: string;
//   query: string;
//   created_at?: string;
// }

// interface ProductRow { ProductID: string; }
// interface ChannelRow { ChannelID: string; }
// interface LocationRow { LocationID: string; }

// interface HistoryRow {
//   ProductID: string;
//   ChannelID: string;
//   LocationID: string;
//   StartDate: string;
//   EndDate: string;
//   Period: string;
//   Qty: number | string;
//   Level?: string;
// }

// interface ForecastRow {
//   ProductID: string;
//   ChannelID: string;
//   LocationID: string;
//   StartDate: string;
//   EndDate: string;
//   Period: string;
//   ForecastQty: number | string;
//   Level?: string;
//   Model?: string;
//   Method?: string;
// }

// interface ForecastApiResponse {
//   variant: string;
//   count: number;
//   rows: ForecastRow[];
// }

// interface SearchKeysResponse {
//   query: string;
//   count: number;
//   keys: Key[];
// }

// // ---- Metrics shape: supports BOTH lowercase and uppercase fields (for existing HTML) ----
// interface Metrics {
//   // canonical (lowercase)
//   n: number;
//   wape: number;
//   mape: number;
//   smape: number;
//   mae: number;
//   rmse: number;
//   bias_pct: number;
//   sae: number;
//   saa: number;

//   // aliases (uppercase) so old HTML like metrics.WAPE compiles
//   WAPE?: number;
//   MAPE?: number;
//   sMAPE?: number;
//   MAE?: number;
//   RMSE?: number;
//   SAE?: number;
//   SAA?: number;
//   BIAS_PCT?: number;
// }

// interface KpiResult {
//   title: string;
//   period: PeriodUI;
//   variant: Variant;
//   lag: number;
//   computedAt: string;

//   // these are referenced by your HTML
//   key?: Key;
//   query?: string;
//   keys?: number;

//   metrics: Metrics;
// }

// @Component({
//   selector: 'app-kpi',
//   standalone: true,
//   imports: [
//     CommonModule,
//     ReactiveFormsModule,
//     HttpClientModule,

//     RowComponent,
//     ColComponent,
//     CardComponent,
//     CardHeaderComponent,
//     CardBodyComponent,
//     ButtonDirective,
//     TableDirective,
//   ],
//   templateUrl: './kpi.component.html',
//   styleUrls: ['./kpi.component.scss'],
// })
// export class KpiComponent implements OnInit, OnDestroy {
//   apiBase = 'http://127.0.0.1:8000';

//   loading = false;
//   errorMessage = '';

//   // dropdowns
//   productIds: string[] = [];
//   channelIds: string[] = [];
//   locationIds: string[] = [];

//   savedSearches: SavedSearch[] = [];

//   // controls used by HTML
//   periodSelection = new FormControl<PeriodUI>('Daily', { nonNullable: true });
//   variantSelection = new FormControl<Variant>('feat', { nonNullable: true });

//   productIdCtrl = new FormControl<string>('', { nonNullable: true });
//   channelIdCtrl = new FormControl<string>('', { nonNullable: true });
//   locationIdCtrl = new FormControl<string>('', { nonNullable: true });

//   savedSearchId = new FormControl<number | null>(null);
//   queryCtrl = new FormControl<string>({ value: '', disabled: true }, { nonNullable: true });

//   // lag used for KPI alignment
//   lagCtrl = new FormControl<number>(1, { nonNullable: true });

//   // results used by HTML
//   singleKpi: KpiResult | null = null;
//   queryKpi: KpiResult | null = null;

//   // KPI cards arrays referenced by HTML
//   singleCards: KpiCard[] = [];
//   queryCards: KpiCard[] = [];
// //   singleCards: Array<{ label: string; value: number; suffix?: string }> = [];
// //   queryCards: Array<{ label: string; value: number; suffix?: string }> = [];

//   private subs: Array<{ unsubscribe: () => void }> = [];

//   constructor(private http: HttpClient) {}

//   ngOnInit(): void {
//     this.loading = true;

//     forkJoin({
//       products: this.http.get<ProductRow[]>(`${this.apiBase}/api/products`),
//       channels: this.http.get<ChannelRow[]>(`${this.apiBase}/api/channels`),
//       locations: this.http.get<LocationRow[]>(`${this.apiBase}/api/locations`),
//       saved: this.http.get<SavedSearch[]>(`${this.apiBase}/api/saved-searches`),
//     })
//       .pipe(
//         map(({ products, channels, locations, saved }) => {
//           this.productIds = (products ?? []).map((r: ProductRow) => String(r.ProductID)).filter(Boolean);
//           this.channelIds = (channels ?? []).map((r: ChannelRow) => String(r.ChannelID)).filter(Boolean);
//           this.locationIds = (locations ?? []).map((r: LocationRow) => String(r.LocationID)).filter(Boolean);
//           this.savedSearches = Array.isArray(saved) ? saved : [];
//           this.refreshQueryPreview();
//           return true;
//         }),
//         finalize(() => (this.loading = false)),
//         catchError((err) => this.fail(err, false))
//       )
//       .subscribe();

//     this.subs.push(
//       this.productIdCtrl.valueChanges.subscribe(() => this.refreshQueryPreview()),
//       this.channelIdCtrl.valueChanges.subscribe(() => this.refreshQueryPreview()),
//       this.locationIdCtrl.valueChanges.subscribe(() => this.refreshQueryPreview()),
//       this.savedSearchId.valueChanges.subscribe((id) => {
//         const s = this.savedSearches.find((x) => x.id === id) ?? null;
//         this.queryCtrl.setValue(s?.query ?? '', { emitEvent: false });
//       })
//     );
//   }

//   ngOnDestroy(): void {
//     for (const s of this.subs) s.unsubscribe();
//   }

//   // -----------------------
//   // template helpers
//   // -----------------------
//   clear(): void {
//     this.errorMessage = '';
//     this.singleKpi = null;
//     this.queryKpi = null;
//     this.singleCards = [];
//     this.queryCards = [];
//   }

//   // -----------------------
//   // internal helpers
//   // -----------------------
//   private fail(err: any, fallback: any) {
//     console.error(err);
//     this.errorMessage = String(err?.error?.detail ?? err?.message ?? 'Unknown error');
//     this.loading = false;
//     return of(fallback);
//   }

//   private getPeriod(): PeriodUI {
//     return (this.periodSelection.value ?? 'Daily') as PeriodUI;
//   }

//   private getVariant(): Variant {
//     return (this.variantSelection.value ?? 'feat') as Variant;
//   }

//   private getSelectedKey(): Key | null {
//     const p = (this.productIdCtrl.value || '').trim();
//     const c = (this.channelIdCtrl.value || '').trim();
//     const l = (this.locationIdCtrl.value || '').trim();
//     if (!p || !c || !l) return null;
//     return { ProductID: p, ChannelID: c, LocationID: l };
//   }

//   private refreshQueryPreview(): void {
//     if (this.savedSearchId.value) return; // keep saved query
//     const key = this.getSelectedKey();
//     if (!key) {
//       this.queryCtrl.setValue('', { emitEvent: false });
//       return;
//     }
//     const q = `productid:${key.ProductID} AND channelid:${key.ChannelID} AND locationid:${key.LocationID}`;
//     this.queryCtrl.setValue(q, { emitEvent: false });
//   }

//   private toDateKey(dateStr: string): string {
//     const s = String(dateStr || '');
//     return s.length >= 10 ? s.slice(0, 10) : s;
//   }

//   private shiftDate(dateStr: string, period: PeriodUI, steps: number): string {
//     const d = new Date(dateStr);
//     if (Number.isNaN(d.getTime())) return this.toDateKey(dateStr);

//     if (period === 'Daily') d.setDate(d.getDate() + steps);
//     else if (period === 'Weekly') d.setDate(d.getDate() + 7 * steps);
//     else d.setMonth(d.getMonth() + steps);

//     const yyyy = d.getFullYear();
//     const mm = String(d.getMonth() + 1).padStart(2, '0');
//     const dd = String(d.getDate()).padStart(2, '0');
//     return `${yyyy}-${mm}-${dd}`;
//   }

//   private emptyMetrics(): Metrics {
//     return { n: 0, wape: 0, mape: 0, smape: 0, mae: 0, rmse: 0, bias_pct: 0, sae: 0, saa: 0 };
//   }

//   private withUpperAliases(m: Metrics): Metrics {
//     return {
//       ...m,
//       WAPE: m.wape,
//       MAPE: m.mape,
//       sMAPE: m.smape,
//       MAE: m.mae,
//       RMSE: m.rmse,
//       SAE: m.sae,
//       SAA: m.saa,
//       BIAS_PCT: m.bias_pct,
//     };
//   }

//   private computeFromPairs(pairs: Array<{ a: number; f: number }>): Metrics {
//     const n = pairs.length;
//     if (!n) return this.emptyMetrics();

//     let sae = 0;
//     let saa = 0;
//     let se2 = 0;
//     let sumAbsPct = 0;
//     let countMape = 0;
//     let sumSmape = 0;
//     let countSmape = 0;
//     let sumBias = 0;

//     for (const { a, f } of pairs) {
//       const e = f - a;
//       const ae = Math.abs(e);

//       sae += ae;
//       saa += Math.abs(a);
//       se2 += e * e;
//       sumBias += e;

//       if (a !== 0) {
//         sumAbsPct += (ae / Math.abs(a)) * 100;
//         countMape++;
//       }

//       const denom = Math.abs(a) + Math.abs(f);
//       if (denom !== 0) {
//         sumSmape += (2 * ae / denom) * 100;
//         countSmape++;
//       }
//     }

//     return {
//       n,
//       sae,
//       saa,
//       mae: sae / n,
//       rmse: Math.sqrt(se2 / n),
//       wape: saa === 0 ? 0 : (sae / saa) * 100,
//       mape: countMape ? sumAbsPct / countMape : 0,
//       smape: countSmape ? sumSmape / countSmape : 0,
//       bias_pct: saa === 0 ? 0 : (sumBias / saa) * 100,
//     };
//   }

//   private computeSingleMetricsWithLag(hist: HistoryRow[], fc: ForecastRow[], lag: number): Metrics {
//     const period = this.getPeriod();

//     const actualByDate = new Map<string, number>();
//     for (const r of hist ?? []) {
//       actualByDate.set(this.toDateKey(r.StartDate), Number(r.Qty ?? 0) || 0);
//     }

//     const pairs: Array<{ a: number; f: number }> = [];
//     for (const r of fc ?? []) {
//       const fcDate = this.toDateKey(r.StartDate);
//       const histDate = this.shiftDate(fcDate, period, -lag);
//       if (!actualByDate.has(histDate)) continue;

//       pairs.push({
//         a: actualByDate.get(histDate) ?? 0,
//         f: Number(r.ForecastQty ?? 0) || 0,
//       });
//     }

//     return this.computeFromPairs(pairs);
//   }

//   private computeQueryMetricsWithLag(hist: HistoryRow[], fc: ForecastRow[], lag: number): Metrics {
//     const period = this.getPeriod();

//     const actualByKeyDate = new Map<string, number>();
//     for (const r of hist ?? []) {
//       const k = `${r.ProductID}|${r.ChannelID}|${r.LocationID}|${this.toDateKey(r.StartDate)}`;
//       actualByKeyDate.set(k, Number(r.Qty ?? 0) || 0);
//     }

//     const pairs: Array<{ a: number; f: number }> = [];
//     for (const r of fc ?? []) {
//       const fcDate = this.toDateKey(r.StartDate);
//       const histDate = this.shiftDate(fcDate, period, -lag);
//       const k = `${r.ProductID}|${r.ChannelID}|${r.LocationID}|${histDate}`;
//       if (!actualByKeyDate.has(k)) continue;

//       pairs.push({
//         a: actualByKeyDate.get(k) ?? 0,
//         f: Number(r.ForecastQty ?? 0) || 0,
//       });
//     }

//     return this.computeFromPairs(pairs);
//   }

//   private buildCards(m: Metrics): KpiCard[] {
//     return [
//       { title: 'n', value: m.n, hint: 'Matched points after lag alignment' },
  
//       { title: 'WAPE', value: m.wape, suffix: '%', hint: 'Sum|err| / Sum|actual| × 100' },
//       { title: 'MAPE', value: m.mape, suffix: '%', hint: 'Mean(|err|/|actual|) × 100 (actual≠0)' },
//       { title: 'sMAPE', value: m.smape, suffix: '%', hint: 'Mean(2|err|/(|a|+|f|)) × 100' },
  
//       { title: 'MAE', value: m.mae, hint: 'Mean absolute error' },
//       { title: 'RMSE', value: m.rmse, hint: 'Root mean squared error' },
  
//       { title: 'Bias', value: m.bias_pct, suffix: '%', hint: 'Sum(err)/Sum|actual| × 100' },
//     ];
//   }
  

//   // -----------------------
//   // API wrappers
//   // -----------------------
//   private loadHistoryByKeys(keys: Key[], limitPerKey: number) {
//     const period = this.getPeriod();
//     const endpoint =
//       period === 'Daily' ? 'daily-by-keys' : period === 'Weekly' ? 'weekly-by-keys' : 'monthly-by-keys';

//     const url = `${this.apiBase}/api/history/${endpoint}`;
//     const params = new HttpParams().set('limit_per_key', String(limitPerKey));

//     return this.http
//       .post<HistoryRow[]>(url, { keys }, { params })
//       .pipe(map((rows) => (Array.isArray(rows) ? rows : [])));
//   }

//   private loadForecastForKey(key: Key, variant: Variant) {
//     const params = new HttpParams()
//       .set('variant', variant)
//       .set('productid', key.ProductID)
//       .set('channelid', key.ChannelID)
//       .set('locationid', key.LocationID)
//       .set('period', this.getPeriod())
//       .set('limit', '5000')
//       .set('offset', '0');

//     return this.http.get<ForecastApiResponse>(`${this.apiBase}/api/forecast`, { params }).pipe(
//       map((res) => (Array.isArray(res?.rows) ? res.rows : []))
//     );
//   }

//   private searchKeysByQuery(q: string) {
//     const params = new HttpParams().set('q', q).set('limit', '50000').set('offset', '0');
//     return this.http.get<SearchKeysResponse>(`${this.apiBase}/api/search`, { params }).pipe(
//       map((res) => (Array.isArray(res?.keys) ? res.keys : []))
//     );
//   }

//   // -----------------------
//   // PUBLIC actions (wire these to your HTML buttons)
//   // -----------------------
//   computeSingleKpi(): void {
//     this.errorMessage = '';
//     this.singleKpi = null;
//     this.singleCards = [];

//     const key = this.getSelectedKey();
//     if (!key) {
//       this.errorMessage = 'Select ProductID, ChannelID and LocationID.';
//       return;
//     }

//     const lag = Math.max(0, Number(this.lagCtrl.value ?? 0));
//     const period = this.getPeriod();
//     const variant = this.getVariant();

//     this.loading = true;

//     this.loadHistoryByKeys([key], 5000)
//       .pipe(
//         switchMap((hist) =>
//           this.loadForecastForKey(key, variant).pipe(map((fc) => ({ hist, fc })))
//         ),
//         finalize(() => (this.loading = false)),
//         catchError((err) => this.fail(err, { hist: [] as HistoryRow[], fc: [] as ForecastRow[] }))
//       )
//       .subscribe(({ hist, fc }) => {
//         const m0 = this.computeSingleMetricsWithLag(hist, fc, lag);
//         const metrics = this.withUpperAliases(m0);

//         this.singleKpi = {
//           title: 'Single KPI',
//           period,
//           variant,
//           lag,
//           computedAt: new Date().toISOString(),
//           key,
//           metrics,
//         };

//         this.singleCards = this.buildCards(metrics);
//       });
//   }

//   computeQueryKpi(): void {
//     this.errorMessage = '';
//     this.queryKpi = null;
//     this.queryCards = [];

//     const q = (this.queryCtrl.value || '').trim();
//     if (!q) {
//       this.errorMessage = 'Choose a saved search or build a query first.';
//       return;
//     }

//     const lag = Math.max(0, Number(this.lagCtrl.value ?? 0));
//     const period = this.getPeriod();
//     const variant = this.getVariant();

//     this.loading = true;

//     this.searchKeysByQuery(q)
//       .pipe(
//         switchMap((keys) => {
//           const limited = keys.slice(0, 200);
//           if (!limited.length) return of({ keys: limited, hist: [] as HistoryRow[], fc: [] as ForecastRow[] });

//           return this.loadHistoryByKeys(limited, 2000).pipe(
//             switchMap((hist) => {
//               const calls = limited.map((k) => this.loadForecastForKey(k, variant));
//               return forkJoin(calls).pipe(
//                 map((lists) => lists.flat()),
//                 map((fc) => ({ keys: limited, hist, fc }))
//               );
//             })
//           );
//         }),
//         finalize(() => (this.loading = false)),
//         catchError((err) => this.fail(err, { keys: [] as Key[], hist: [] as HistoryRow[], fc: [] as ForecastRow[] }))
//       )
//       .subscribe(({ keys, hist, fc }) => {
//         const m0 = this.computeQueryMetricsWithLag(hist, fc, lag);
//         const metrics = this.withUpperAliases(m0);

//         this.queryKpi = {
//           title: 'Query KPI',
//           period,
//           variant,
//           lag,
//           computedAt: new Date().toISOString(),
//           query: q,
//           keys: keys.length,
//           metrics,
//         };

//         this.queryCards = this.buildCards(metrics);
//       });
//   }
// }
import { CommonModule } from '@angular/common';
import { HttpClient, HttpClientModule, HttpParams } from '@angular/common/http';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormControl, ReactiveFormsModule } from '@angular/forms';
import { catchError, finalize, forkJoin, map, of, switchMap } from 'rxjs';

// CoreUI standalone imports
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
type Variant = 'baseline' | 'feat'; // kept for UI, not used in history-only KPI

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

interface SearchKeysResponse {
  query: string;
  count: number;
  keys: Key[];
}

// ---- Metrics shape: supports BOTH lowercase and uppercase fields (for existing HTML) ----
interface Metrics {
  // canonical (lowercase)
  n: number;
  wape: number;
  mape: number;
  smape: number;
  mae: number;
  rmse: number;
  bias_pct: number;
  sae: number;
  saa: number;

  // aliases (uppercase)
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
  variant: Variant;     // shown in UI (even though history-only KPI)
  lag: number;
  computedAt: string;

  // referenced by HTML
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
  apiBase = 'http://127.0.0.1:8000';

  loading = false;
  errorMessage = '';

  // dropdowns
  productIds: string[] = [];
  channelIds: string[] = [];
  locationIds: string[] = [];

  savedSearches: SavedSearch[] = [];

  // controls used by HTML
  periodSelection = new FormControl<PeriodUI>('Daily', { nonNullable: true });
  variantSelection = new FormControl<Variant>('feat', { nonNullable: true }); // UI only

  productIdCtrl = new FormControl<string>('', { nonNullable: true });
  channelIdCtrl = new FormControl<string>('', { nonNullable: true });
  locationIdCtrl = new FormControl<string>('', { nonNullable: true });

  savedSearchId = new FormControl<number | null>(null);

  // ✅ make it editable (your HTML uses an <input>)
  queryCtrl = new FormControl<string>('', { nonNullable: true });

  // lag used for KPI alignment
  lagCtrl = new FormControl<number>(1, { nonNullable: true });

  // results used by HTML
  singleKpi: KpiResult | null = null;
  queryKpi: KpiResult | null = null;

  // KPI cards arrays referenced by HTML
  singleCards: KpiCard[] = [];
  queryCards: KpiCard[] = [];

  private subs: Array<{ unsubscribe: () => void }> = [];

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.loading = true;

    forkJoin({
      products: this.http.get<ProductRow[]>(`${this.apiBase}/api/products`),
      channels: this.http.get<ChannelRow[]>(`${this.apiBase}/api/channels`),
      locations: this.http.get<LocationRow[]>(`${this.apiBase}/api/locations`),
      saved: this.http.get<SavedSearch[]>(`${this.apiBase}/api/saved-searches`),
    })
      .pipe(
        map(({ products, channels, locations, saved }) => {
          this.productIds = (products ?? []).map((r) => String(r.ProductID)).filter(Boolean);
          this.channelIds = (channels ?? []).map((r) => String(r.ChannelID)).filter(Boolean);
          this.locationIds = (locations ?? []).map((r) => String(r.LocationID)).filter(Boolean);
          this.savedSearches = Array.isArray(saved) ? saved : [];
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

  // -----------------------
  // template helpers
  // -----------------------
  clear(): void {
    this.errorMessage = '';
    this.singleKpi = null;
    this.queryKpi = null;
    this.singleCards = [];
    this.queryCards = [];
  }

  // -----------------------
  // internal helpers
  // -----------------------
  private fail(err: any, fallback: any) {
    console.error(err);
    this.errorMessage = String(err?.error?.detail ?? err?.message ?? 'Unknown error');
    this.loading = false;
    return of(fallback);
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
    if (this.savedSearchId.value) return; // keep saved query
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

  // shift date by lag steps in the chosen grain
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

  // ✅ HISTORY-ONLY KPI: forecast = actual(t-lag)
  private computeSingleHistoryLagKpi(hist: HistoryRow[], lag: number): Metrics {
    const period = this.getPeriod();

    const actualByDate = new Map<string, number>();
    for (const r of hist ?? []) {
      actualByDate.set(this.toDateKey(r.StartDate), Number(r.Qty ?? 0) || 0);
    }

    // For each date d in history, compare actual(d) vs actual(d-lag)
    const pairs: Array<{ a: number; f: number }> = [];
    for (const [d, a] of actualByDate.entries()) {
      const lagDate = this.shiftDate(d, period, -lag);
      if (!actualByDate.has(lagDate)) continue;
      const f = actualByDate.get(lagDate) ?? 0;
      pairs.push({ a, f });
    }

    return this.computeFromPairs(pairs);
  }

  // ✅ HISTORY-ONLY KPI for query: per key+date
  private computeQueryHistoryLagKpi(hist: HistoryRow[], lag: number): Metrics {
    const period = this.getPeriod();

    const actualByKeyDate = new Map<string, number>();
    for (const r of hist ?? []) {
      const k = `${r.ProductID}|${r.ChannelID}|${r.LocationID}|${this.toDateKey(r.StartDate)}`;
      actualByKeyDate.set(k, Number(r.Qty ?? 0) || 0);
    }

    const pairs: Array<{ a: number; f: number }> = [];
    for (const [k, a] of actualByKeyDate.entries()) {
      // key = P|C|L|date
      const parts = k.split('|');
      if (parts.length !== 4) continue;

      const [p, c, l, d] = parts;
      const lagDate = this.shiftDate(d, period, -lag);
      const kLag = `${p}|${c}|${l}|${lagDate}`;
      if (!actualByKeyDate.has(kLag)) continue;

      const f = actualByKeyDate.get(kLag) ?? 0;
      pairs.push({ a, f });
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

  // -----------------------
  // API wrappers (history + search only)
  // -----------------------
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

  private searchKeysByQuery(q: string) {
    const params = new HttpParams().set('q', q).set('limit', '50000').set('offset', '0');
    return this.http.get<SearchKeysResponse>(`${this.apiBase}/api/search`, { params }).pipe(
      map((res) => (Array.isArray(res?.keys) ? res.keys : []))
    );
  }

  // -----------------------
  // PUBLIC actions (wired to your HTML buttons)
  // -----------------------
  computeSingleKpi(): void {
    this.errorMessage = '';
    this.singleKpi = null;
    this.singleCards = [];

    const key = this.getSelectedKey();
    if (!key) {
      this.errorMessage = 'Select ProductID, ChannelID and LocationID.';
      return;
    }

    const lag = Math.max(1, Number(this.lagCtrl.value ?? 1)); // lag >= 1
    const period = this.getPeriod();
    const variant = this.getVariant(); // UI only

    this.loading = true;

    this.loadHistoryByKeys([key], 5000)
      .pipe(
        finalize(() => (this.loading = false)),
        catchError((err) => this.fail(err, [] as HistoryRow[]))
      )
      .subscribe((hist) => {
        const m0 = this.computeSingleHistoryLagKpi(hist, lag);
        const metrics = this.withUpperAliases(m0);

        if (metrics.n === 0) {
          this.errorMessage = `Not enough history points to compute lag=${lag} KPI (need at least lag+1 points).`;
        }

        this.singleKpi = {
          title: 'Single KPI (History lag)',
          period,
          variant,
          lag,
          computedAt: new Date().toISOString(),
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

    const lag = Math.max(1, Number(this.lagCtrl.value ?? 1));
    const period = this.getPeriod();
    const variant = this.getVariant(); // UI only

    this.loading = true;

    this.searchKeysByQuery(q)
      .pipe(
        switchMap((keys) => {
          const limited = keys.slice(0, 200);
          if (!limited.length) return of({ keys: limited, hist: [] as HistoryRow[] });

          return this.loadHistoryByKeys(limited, 2000).pipe(
            map((hist) => ({ keys: limited, hist }))
          );
        }),
        finalize(() => (this.loading = false)),
        catchError((err) => this.fail(err, { keys: [] as Key[], hist: [] as HistoryRow[] }))
      )
      .subscribe(({ keys, hist }) => {
        const m0 = this.computeQueryHistoryLagKpi(hist, lag);
        const metrics = this.withUpperAliases(m0);

        if (metrics.n === 0) {
          this.errorMessage = `Not enough history points to compute lag=${lag} KPI for this query.`;
        }

        this.queryKpi = {
          title: 'Query KPI (History lag)',
          period,
          variant,
          lag,
          computedAt: new Date().toISOString(),
          query: q,
          keys: keys.length,
          metrics,
        };

        this.queryCards = this.buildCards(metrics);
      });
  }
}
