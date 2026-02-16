import { CommonModule } from '@angular/common';
import { Component, OnInit, inject } from '@angular/core';
import { FormBuilder, FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { HttpClient, HttpClientModule, HttpParams } from '@angular/common/http';
import {
  CardComponent, CardHeaderComponent, CardBodyComponent, CardFooterComponent,
  RowComponent, ColComponent, ButtonDirective, TextColorDirective, TableDirective
} from '@coreui/angular';
import { IconDirective } from '@coreui/icons-angular';
import { firstValueFrom } from 'rxjs';

/** --- Backend contracts --- */
interface KeyTriplet { ProductID: string; ChannelID: string; LocationID: string; }
interface SearchResult { query: string; count: number; keys: KeyTriplet[]; }
interface SavedSearch { id?: number; name: string; query: string; created_at?: string; }

interface ClassificationRow {
  ProductID: string;
  ChannelID: string;
  LocationID: string;

  periods: number;
  nonzero_count: number;
  adi: number | null;
  cv2: number | null;

  category: 'Smooth' | 'Intermittent' | 'Erratic' | 'Sparse' | 'NotEnoughHistory';
  seasonal?: boolean | null;

  createdAt?: string;
}

interface HistoryRow {
  ProductID: string;
  ChannelID: string;
  LocationID: string;
  StartDate: string;
  Qty: number;
}

type PeriodView = 'Daily' | 'Weekly' | 'Monthly';
type Algo = 'HoltWinters' | 'XGBoost' | 'MovingAverage' | 'Croston' | 'ARIMA' | 'ETS';

@Component({
  standalone: true,
  selector: 'app-classify-forecast-elements',
  templateUrl: './classify-forecast-elements.component.html',
  styleUrls: ['./classify-forecast-elements.component.scss'],
  imports: [
    CommonModule, ReactiveFormsModule, HttpClientModule,
    TextColorDirective, TableDirective,
    CardComponent, CardHeaderComponent, CardBodyComponent, CardFooterComponent,
    RowComponent, ColComponent, ButtonDirective, IconDirective
  ]
})
export class ClassifyForecastElementsComponent implements OnInit {
  /** Services */
  private http = inject(HttpClient);
  private fb   = inject(FormBuilder);

  /** API base */
  readonly API = 'http://127.0.0.1:8000/api';

  /** Saved searches */
  savedSearches: SavedSearch[] = [];
  selectedSavedIndex = new FormControl<number>(-1, { nonNullable: true });
  periodSelection   = new FormControl<PeriodView>('Daily', { nonNullable: true });

  /** UI state */
  loading = false;
  errorMsg: string | null = null;

  /** Save state */
  saving = false;
  saveMsg: string | null = null;
  saveErr: string | null = null;

  /** Results */
  rows: ClassificationRow[] = [];

  /** Settings form */
  form: FormGroup = this.fb.group({
    autoEnabled:   this.fb.control<boolean>(true,  { nonNullable: true }),
    manualEnabled: this.fb.control<boolean>(false, { nonNullable: true }),

    autoLookback:  this.fb.control<number>(12, { nonNullable: true, validators: [Validators.min(1)] }),
    autoAlgo:      this.fb.control<Algo>('XGBoost', { nonNullable: true }),

    manual: this.fb.group({
      Smooth:           this.fb.control<Algo>('HoltWinters',   { nonNullable: true }),
      Intermittent:     this.fb.control<Algo>('XGBoost',       { nonNullable: true }),
      Erratic:          this.fb.control<Algo>('XGBoost',       { nonNullable: true }),
      Sparse:           this.fb.control<Algo>('MovingAverage', { nonNullable: true }),
      NotEnoughHistory: this.fb.control<Algo>('XGBoost',       { nonNullable: true }),
    })
  });

  readonly algoOptions: Algo[] = ['XGBoost','HoltWinters','MovingAverage','Croston','ARIMA','ETS'];

  ngOnInit(): void {
    this.refreshSavedSearches();

    // keep auto/manual exclusive
    this.form.get('autoEnabled')!.valueChanges.subscribe(v => {
      if (v) this.form.get('manualEnabled')!.setValue(false, { emitEvent: false });
      else if (!this.form.get('manualEnabled')!.value) this.form.get('manualEnabled')!.setValue(true, { emitEvent: false });
    });
    this.form.get('manualEnabled')!.valueChanges.subscribe(v => {
      if (v) this.form.get('autoEnabled')!.setValue(false, { emitEvent: false });
      else if (!this.form.get('autoEnabled')!.value) this.form.get('autoEnabled')!.setValue(true, { emitEvent: false });
    });
  }

  refreshSavedSearches() {
    this.http.get<SavedSearch[]>(`${this.API}/saved-searches`).subscribe({
      next: rows => this.savedSearches = rows || [],
      error: ()   => this.savedSearches = []
    });
  }

  /** period label → backend slug */
  private periodSlug(): 'daily' | 'weekly' | 'monthly' {
    const p = this.periodSelection.value;
    return p === 'Weekly' ? 'weekly' : p === 'Monthly' ? 'monthly' : 'daily';
  }

  /** What algorithm will be used for a category (for display & saving) */
  chosenAlgo(category: ClassificationRow['category']): string {
    if (this.form.get('manualEnabled')!.value) {
      const m = this.form.get('manual')!.value as Record<string, Algo>;
      return m[category] ?? '(none)';
    }
    return this.form.get('autoAlgo')!.value as string;
  }

  trackByKey = (_: number, r: ClassificationRow) =>
    `${r.ProductID}||${r.ChannelID}||${r.LocationID}`;

  /** Run: resolve keys from saved search then classify & load results with ADI/CV² */
  async runClassification() {
    this.errorMsg = null;
    this.saveMsg = this.saveErr = null;
    this.rows = [];

    const idx = this.selectedSavedIndex.value;
    if (idx < 0 || idx >= this.savedSearches.length) {
      this.errorMsg = 'Please select a saved search.';
      return;
    }

    const saved = this.savedSearches[idx];
    const q = (saved.query || '').trim();
    if (!q) {
      this.errorMsg = 'Saved search query is empty.';
      return;
    }

    this.loading = true;
    try {
      // 1) Resolve keys from /api/search
      const params = new HttpParams().set('q', q).set('limit', 20000).set('offset', 0);
      const sr = await firstValueFrom(this.http.get<SearchResult>(`${this.API}/search`, { params }));
      const keys = sr?.keys ?? [];
      if (!keys.length) {
        this.errorMsg = 'No matches for this saved query.';
        return;
      }

      // 2) Trigger backend classification compute (kept as-is)
      const computePayload: any = {
        period: this.periodSlug(),
        lookback_buckets: 8,
        min_sum: 1.0,
      };
      await firstValueFrom(this.http.post(`${this.API}/classify/compute`, computePayload));

      // 3) Fetch classification results
      interface BackendClassRow {
        ProductID: string;
        ChannelID: string;
        LocationID: string;
        Period: string;
        Label: string;
        Score: number;
        IsActive: boolean;
        ComputedAt: string;
      }

      const resParams = new HttpParams()
        .set('period', this.periodSlug())
        .set('include_inactive', 'true');

      const allRes = await firstValueFrom(
        this.http.get<BackendClassRow[]>(`${this.API}/classify/results`, { params: resParams })
      );

      if (!allRes?.length) {
        this.errorMsg = 'No classification results found. Please run Cleanse History first.';
        return;
      }

      // Restrict to saved-search keys
      const keySet = new Set(keys.map(k => `${k.ProductID}||${k.ChannelID}||${k.LocationID}`));
      const filtered = (allRes || []).filter(r => keySet.has(`${r.ProductID}||${r.ChannelID}||${r.LocationID}`));

      if (!filtered.length) {
        this.errorMsg = 'No Cleansed-History found for these keys. Please run Cleanse History first.';
        return;
      }

      // 4) Load history for these keys (used to compute ADI/CV²)
      const histRows = await firstValueFrom(
        this.http.post<HistoryRow[]>(
          `${this.API}/history/${this.periodSlug()}-by-keys`,
          { keys }
        )
      );

      // Build per-key series
      const seriesMap = new Map<string, number[]>();
      for (const h of histRows || []) {
        const key = `${h.ProductID}||${h.ChannelID}||${h.LocationID}`;
        const qty = Number(h.Qty) || 0;
        if (!seriesMap.has(key)) seriesMap.set(key, []);
        seriesMap.get(key)!.push(qty);
      }

      const minNonZero = 6;

      function computeMetrics(qtys: number[]): {
        periods: number;
        nonzero: number;
        adi: number | null;
        cv2: number | null;
        category: ClassificationRow['category'];
      } {
        const periods = qtys.length;
        const nonzeroVals = qtys.filter(q => q > 0);
        const nonzero = nonzeroVals.length;

        if (!periods || nonzero < minNonZero) {
          return { periods, nonzero, adi: null, cv2: null, category: 'NotEnoughHistory' };
        }

        const adi = periods / nonzero;

        const mean = nonzeroVals.reduce((s, q) => s + q, 0) / nonzeroVals.length;

        if (!(mean > 0) || nonzeroVals.length < 2) {
          return { periods, nonzero, adi, cv2: null, category: 'NotEnoughHistory' };
        }

        const variance =
          nonzeroVals.reduce((s, q) => s + (q - mean) * (q - mean), 0) /
          (nonzeroVals.length - 1);

        const std = Math.sqrt(variance);
        const cv2 = (std / mean) * (std / mean);

        let category: ClassificationRow['category'] = 'Sparse';
        if (adi < 1.32 && cv2 < 0.49) category = 'Smooth';
        else if (adi < 1.32 && cv2 >= 0.49) category = 'Erratic';
        else if (adi >= 1.32 && cv2 < 0.49) category = 'Intermittent';
        else category = 'Sparse';

        return { periods, nonzero, adi, cv2, category };
      }

      // 5) Build UI rows
      this.rows = filtered.map(r => {
        const key = `${r.ProductID}||${r.ChannelID}||${r.LocationID}`;
        const qtys = seriesMap.get(key) || [];
        const m = computeMetrics(qtys);

        return {
          ProductID: r.ProductID,
          ChannelID: r.ChannelID,
          LocationID: r.LocationID,
          periods: m.periods,
          nonzero_count: m.nonzero,
          adi: m.adi,
          cv2: m.cv2,
          category: m.category,
          seasonal: null,
          createdAt: r.ComputedAt,
        };
      });

      if (!this.rows.length) {
        this.errorMsg = 'Run succeeded but produced 0 rows after filtering.';
      }
    } catch (e: any) {
      this.errorMsg = e?.error?.detail || e?.message || 'Failed to classify.';
    } finally {
      this.loading = false;
    }
  }

  /** Save computed rows to the backend so the Data page can read them */
  async saveResults() {
    this.saveMsg = this.saveErr = null;

    if (!this.rows.length) {
      this.saveErr = 'Nothing to save. Run classification first.';
      return;
    }

    const payload = {
      period: this.periodSlug(),
      rows: this.rows.map(r => ({
        ProductID: r.ProductID,
        ChannelID: r.ChannelID,
        LocationID: r.LocationID,
        ADI: r.adi,
        CV2: r.cv2,
        Category: r.category,
        Algorithm: this.chosenAlgo(r.category),
        CreatedAt: r.createdAt ?? null,
      }))
    };

    this.saving = true;
    try {
      const res = await firstValueFrom(
        this.http.post<{ ok: boolean; count: number }>(`${this.API}/classify/save`, payload)
      );
      this.saveMsg = `Saved ${res?.count ?? payload.rows.length} result(s).`;
    } catch (e: any) {
      this.saveErr = e?.error?.detail || e?.message || 'Failed to save results.';
    } finally {
      this.saving = false;
    }
  }
}
