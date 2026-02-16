import { CommonModule } from '@angular/common';
import { Component, OnInit, inject, computed, signal } from '@angular/core';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule, FormControl } from '@angular/forms';
import { HttpClient, HttpClientModule, HttpParams } from '@angular/common/http';
import {
  ButtonDirective, CardBodyComponent, CardComponent, CardFooterComponent, CardHeaderComponent,
  ColComponent, RowComponent, TextColorDirective
} from '@coreui/angular';
import { IconDirective } from '@coreui/icons-angular';
import { firstValueFrom } from 'rxjs';

/* ---------- Types ---------- */
type OutlierMethod = 'Z-Score' | 'IQR' | 'MAD' | 'Hampel';
type MissingMethod = 'Drop Rows' | 'Forward Fill' | 'Backward Fill' | 'Zero' | 'Mean' | 'Median' | 'Linear Interpolate';
type PeriodUi = 'Daily' | 'Weekly' | 'Monthly';
type PeriodLiteral = 'daily' | 'weekly' | 'monthly';

interface CleanseProfile {
  id?: number;
  name: string;
  config: any;
  created_at?: string;
}

interface SavedSearch { id: number; name: string; query: string; created_at?: string; }
interface KeyTriplet { ProductID: string; ChannelID: string; LocationID: string; }
interface SearchResult { query: string; count: number; keys: KeyTriplet[]; }

interface HistoryRow {
  ProductID: string;
  ChannelID: string;
  LocationID: string;
  StartDate: string;
  EndDate: string;
  Qty: number | null;   // treat null/NaN as missing in cleanse pipeline
  Level: string;
  Period: string;
  Type: string;         // Normal-History or Cleansed-History, etc.
}

@Component({
  standalone: true,
  selector: 'app-cleanse-history',
  templateUrl: './cleanse-history.component.html',
  styleUrls: ['./cleanse-history.component.scss'],
  imports: [
    CommonModule, ReactiveFormsModule, HttpClientModule,
    TextColorDirective,
    CardComponent, CardHeaderComponent, CardBodyComponent, CardFooterComponent,
    RowComponent, ColComponent, ButtonDirective, IconDirective
  ]
})
export class CleanseHistoryComponent implements OnInit {
  private _http = inject(HttpClient);
  private _fb = inject(FormBuilder);

  readonly API = 'http://127.0.0.1:8000/api';

  /* ---------- Saved searches & period ---------- */
  savedSearches: SavedSearch[] = [];
  selectedSavedIndex = new FormControl<number>(-1, { nonNullable: true });
  periodSelection = new FormControl<PeriodUi>('Daily', { nonNullable: true });

  /* ---------- UI state ---------- */
  errorMessage: string | null = null;
  loading = false;

  /* ---------- Cleanse form ---------- */
  form: FormGroup = this._fb.group({
    profileName: this._fb.control<string>('', { validators: [Validators.required], nonNullable: true }),

    outlierMethod: this._fb.control<OutlierMethod>('Z-Score', { nonNullable: true }),
    outlierParams: this._fb.group({
      zThreshold: this._fb.control<number>(3, { nonNullable: true }),
      iqrK: this._fb.control<number>(1.5, { nonNullable: true }),
      madK: this._fb.control<number>(3, { nonNullable: true }),
      hampelWindow: this._fb.control<number>(7, { nonNullable: true }),
      hampelK: this._fb.control<number>(3, { nonNullable: true }),
    }),

    missingMethod: this._fb.control<MissingMethod>('Forward Fill', { nonNullable: true }),
    missingParams: this._fb.group({
      interpolateOrder: this._fb.control<number>(1, { nonNullable: true }),
      fillConstant: this._fb.control<number>(0, { nonNullable: true }),
    }),
  });

  get outlierMethod(): OutlierMethod {
    return this.form.get('outlierMethod')!.value as OutlierMethod;
  }

  /* ---------- Data the cleanse runs on ---------- */
  rawHistory: HistoryRow[] = [];     // fetched from /history/{period}-by-keys for a saved search
  previewRows: HistoryRow[] = [];    // cleansed rows ready to save

  /* ---------- Save status ---------- */
  saving = false;
  saveMessage: string | null = null;
  saveError: string | null = null;

  ngOnInit(): void {
    this.refreshProfiles();
    this.refreshSavedSearches();
  }

  /* =========================================================
   * Saved Profiles (load/save settings only)
   * =======================================================*/
  profiles: CleanseProfile[] = [];

  refreshProfiles() {
    this._http.get<CleanseProfile[]>(`${this.API}/cleanse/profiles`).subscribe({
      next: rows => (this.profiles = rows || []),
      error: e => console.error('Failed to load profiles', e)
    });
  }

  loadProfile(p: CleanseProfile) {
    if (!p) return;
    const { name, config } = p;
    this.form.patchValue({
      profileName: name || '',
      outlierMethod: config?.outlierMethod ?? 'Z-Score',
      outlierParams: {
        zThreshold: config?.outlierParams?.zThreshold ?? 3,
        iqrK: config?.outlierParams?.iqrK ?? 1.5,
        madK: config?.outlierParams?.madK ?? 3,
        hampelWindow: config?.outlierParams?.hampelWindow ?? 7,
        hampelK: config?.outlierParams?.hampelK ?? 3,
      },
      missingMethod: config?.missingMethod ?? 'Forward Fill',
      missingParams: {
        interpolateOrder: config?.missingParams?.interpolateOrder ?? 1,
        fillConstant: config?.missingParams?.fillConstant ?? 0,
      }
    });
  }

  saveProfile() {
    this.errorMessage = null;
    if (this.form.invalid) {
      this.errorMessage = 'Please enter a profile name.';
      this.form.markAllAsTouched();
      return;
    }
    const { profileName, ...config } = this.form.getRawValue() as any;
    const body = { name: profileName, config };

    this.loading = true;
    this._http.post(`${this.API}/cleanse/profiles`, body).subscribe({
      next: () => {
        this.loading = false;
        this.refreshProfiles();
      },
      error: (e) => {
        this.loading = false;
        this.errorMessage = e?.error?.detail || 'Failed to save profile.';
      }
    });
  }

  /* =========================================================
   * Saved Search → Keys → History (the subset we cleanse)
   * =======================================================*/
  refreshSavedSearches() {
    this._http.get<SavedSearch[]>(`${this.API}/saved-searches`).subscribe({
      next: rows => (this.savedSearches = rows || []),
      error: e => console.error('Failed to load saved searches', e)
    });
  }

  private periodSlug(): PeriodLiteral {
    const p = this.periodSelection.value;
    return p === 'Weekly' ? 'weekly' : p === 'Monthly' ? 'monthly' : 'daily';
  }

  async loadHistoryFromSaved() {
    this.errorMessage = null;
    this.rawHistory = [];
    this.previewRows = [];

    const idx = this.selectedSavedIndex.value ?? -1;
    if (idx < 0 || idx >= this.savedSearches.length) {
      this.errorMessage = 'Please choose a saved search.';
      return;
    }
    const q = this.savedSearches[idx].query;

    try {
      this.loading = true;

      // 1) resolve keys via /search
      const params = new HttpParams().set('q', q).set('limit', 20000).set('offset', 0);
      const search = await firstValueFrom(this._http.get<SearchResult>(`${this.API}/search`, { params }));
      const keys = search?.keys ?? [];
      if (!keys.length) {
        this.errorMessage = 'No matches for this saved query.';
        return;
      }

      // 2) fetch history by keys for the selected period
      const endpoint = `${this.API}/history/${this.periodSlug()}-by-keys`;
      const rows = await firstValueFrom(this._http.post<HistoryRow[]>(endpoint, { keys }));

      // normalize Qty to number|null
      this.rawHistory = (rows || []).map(r => ({
        ...r,
        Qty: (r.Qty === null || r.Qty === undefined || isNaN(Number(r.Qty))) ? null : Number(r.Qty),
      }));

    } catch (e: any) {
      this.errorMessage = e?.error?.detail || e?.message || 'Failed to load history for saved search.';
    } finally {
      this.loading = false;
    }
  }

  /* =========================================================
   * Cleanse preview (outliers + missing) on loaded history
   * =======================================================*/
  async generatePreview() {
    this.saveMessage = null;
    this.saveError = null;
    this.errorMessage = null;

    if (!this.rawHistory.length) {
      this.errorMessage = 'Load history using a saved search first.';
      return;
    }

    // group rows by key and sort by StartDate (ascending)
    const byKey = new Map<string, HistoryRow[]>();
    for (const r of this.rawHistory) {
      const k = `${r.ProductID}||${r.ChannelID}||${r.LocationID}`;
      if (!byKey.has(k)) byKey.set(k, []);
      byKey.get(k)!.push(r);
    }
    for (const rows of byKey.values()) {
      rows.sort((a, b) => (a.StartDate < b.StartDate ? -1 : a.StartDate > b.StartDate ? 1 : 0));
    }

    // read settings
    const outlierMethod = this.form.get('outlierMethod')!.value as OutlierMethod;
    const op = this.form.get('outlierParams')!.value as any;
    const missingMethod = this.form.get('missingMethod')!.value as MissingMethod;
    const mp = this.form.get('missingParams')!.value as any;

    const cleaned: HistoryRow[] = [];

    for (const rows of byKey.values()) {
      // get a working copy of the series
      const series = rows.map(r => r.Qty);

      // ---- Outliers (winsorize to bounds for visibility/preservation) ----
      const newSeries = this.applyOutliers(series, outlierMethod, op);

      // ---- Missing values ----
      const finalSeries = this.applyMissing(newSeries, missingMethod, mp);

      // materialize new rows with Type= C* (but we keep original for reference)
      for (let i = 0; i < rows.length; i++) {
        const base = rows[i];
        const q = finalSeries[i];
        if (q === null && missingMethod === 'Drop Rows') {
          // skip row when dropping
          continue;
        }
        cleaned.push({
          ...base,
          Qty: q ?? 0, // fallback 0 if still null (shouldn’t happen except weird edges)
          Type: 'Cleansed-History'
        });
      }
    }

    this.previewRows = cleaned;
  }

  /* ---------- Outlier handlers ---------- */
  private applyOutliers(series: (number | null)[], method: OutlierMethod, op: any): (number | null)[] {
    // ignore null for stats; keep indexes
    const vals = series.map(v => (v === null ? NaN : Number(v)));
    const numbers = vals.filter(v => !isNaN(v));
    if (numbers.length < 2) return [...series];

    const clip = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

    if (method === 'Z-Score') {
      const z = Number(op?.zThreshold ?? 3);
      const mean = numbers.reduce((a, b) => a + b, 0) / numbers.length;
      const varv = numbers.reduce((a, b) => a + (b - mean) ** 2, 0) / (numbers.length - 1);
      const sd = Math.sqrt(varv || 0);
      if (!sd) return [...series];
      const lo = mean - z * sd, hi = mean + z * sd;
      return vals.map(v => (isNaN(v) ? null : clip(v, lo, hi)));
    }

    if (method === 'IQR') {
      const k = Number(op?.iqrK ?? 1.5);
      const sorted = [...numbers].sort((a, b) => a - b);
      const q1 = quantile(sorted, 0.25);
      const q3 = quantile(sorted, 0.75);
      const iqr = q3 - q1;
      const lo = q1 - k * iqr, hi = q3 + k * iqr;
      return vals.map(v => (isNaN(v) ? null : clip(v, lo, hi)));
    }

    if (method === 'MAD') {
      const k = Number(op?.madK ?? 3);
      const med = median(numbers);
      const mad = median(numbers.map(x => Math.abs(x - med))) || 0;
      const sigma = 1.4826 * mad;
      if (!sigma) return [...series];
      const lo = med - k * sigma, hi = med + k * sigma;
      return vals.map(v => (isNaN(v) ? null : clip(v, lo, hi)));
    }

    // Hampel (rolling)
    if (method === 'Hampel') {
      const w = Math.max(1, Number(op?.hampelWindow ?? 7));
      const k = Number(op?.hampelK ?? 3);
      const out = vals.slice();
      for (let i = 0; i < vals.length; i++) {
        if (isNaN(vals[i])) continue;
        const s = Math.max(0, i - Math.floor(w / 2));
        const e = Math.min(vals.length, s + w);
        const win = vals.slice(s, e).filter(v => !isNaN(v));
        if (win.length < 2) continue;
        const m = median(win);
        const localMad = median(win.map(x => Math.abs(x - m))) || 0;
        const sigma = 1.4826 * localMad;
        if (!sigma) continue;
        const lo = m - k * sigma, hi = m + k * sigma;
        out[i] = clip(vals[i], lo, hi);
      }
      return out.map(v => (isNaN(v) ? null : v));
    }

    return [...series];
  }

  /* ---------- Missing handlers ---------- */
  private applyMissing(series: (number | null)[], method: MissingMethod, mp: any): (number | null)[] {
    const out = [...series];

    if (method === 'Drop Rows') {
      // caller will drop rows whose value is null
      return out.map(v => (v === null ? null : v));
    }

    if (method === 'Zero') {
      return out.map(v => (v === null ? 0 : v));
    }

    if (method === 'Forward Fill') {
      let last: number | null = null;
      for (let i = 0; i < out.length; i++) {
        if (out[i] === null) out[i] = last;
        else last = out[i];
      }
      return out;
    }

    if (method === 'Backward Fill') {
      let next: number | null = null;
      for (let i = out.length - 1; i >= 0; i--) {
        if (out[i] === null) out[i] = next;
        else next = out[i];
      }
      return out;
    }

    if (method === 'Mean') {
      const vals = out.filter(v => v !== null) as number[];
      const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
      return out.map(v => (v === null ? mean : v));
    }

    if (method === 'Median') {
      const vals = out.filter(v => v !== null) as number[];
      const med = vals.length ? median(vals) : 0;
      return out.map(v => (v === null ? med : v));
    }

    if (method === 'Linear Interpolate') {
      // simple linear interpolation for inner gaps; edges left as-is (you can FF/BF afterward if desired)
      let i = 0;
      while (i < out.length) {
        if (out[i] !== null) { i++; continue; }
        const start = i - 1;
        while (i < out.length && out[i] === null) i++;
        const end = i; // first non-null after gap
        const y0 = start >= 0 ? out[start] : null;
        const y1 = end < out.length ? out[end] : null;
        if (y0 !== null && y1 !== null) {
          const span = end - start;
          for (let k = 1; k < span; k++) {
            out[start + k] = y0 + (k / span) * (y1 - y0);
          }
        }
      }
      return out;
    }

    return out;
  }

  /* =========================================================
   * Save cleansed rows (UPSERT by unique key incl. Type)
   * =======================================================*/
  private mapToPayloadRow(r: HistoryRow) {
    const start = (r.StartDate || '').trim();
    const endRaw = (r.EndDate ?? '').toString().trim();
  
    // Treat '', 'None', 'null', 'undefined' as missing
    const endMissing =
      !endRaw ||
      endRaw.toLowerCase() === 'none' ||
      endRaw.toLowerCase() === 'null' ||
      endRaw.toLowerCase() === 'undefined';
  
    const end = endMissing ? start : endRaw;
  
    return {
      ProductID: r.ProductID,
      ChannelID: r.ChannelID,
      LocationID: r.LocationID,
      StartDate: start,
      EndDate: end,
      Qty: Number(r.Qty ?? 0),
      Level: String(r.Level ?? ''),
      NetPrice: Number((r as any).NetPrice ?? 0),
      ListPrice: Number((r as any).ListPrice ?? 0),

    };
  }


  async saveCleansedHistory() {
    this.saveMessage = null;
    this.saveError = null;

    if (!this.previewRows?.length) {
      this.saveError = 'No cleansed rows to save. Generate a preview first.';
      return;
    }

    const rows = this.previewRows.map(r => this.mapToPayloadRow(r));
    const body = { period: this.periodSlug(), rows }; // backend defaults Type='Cleansed-History'

    try {
      this.saving = true;
      const res = await firstValueFrom(
        this._http.post<{ ok: boolean; count: number }>(`${this.API}/history/ingest-cleansed`, body)
      );
      this.saveMessage = `Saved ${res?.count ?? rows.length} cleansed rows.`;
    } catch (e: any) {
      this.saveError = e?.error?.detail || e?.message || 'Failed to save cleansed history.';
    } finally {
      this.saving = false;
    }
  }
}

/* ---------- tiny stats helpers ---------- */
function median(arr: number[]): number {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function quantile(sortedAsc: number[], q: number): number {
  if (!sortedAsc.length) return 0;
  const pos = (sortedAsc.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sortedAsc[base + 1] !== undefined) {
    return sortedAsc[base] + rest * (sortedAsc[base + 1] - sortedAsc[base]);
  }
  return sortedAsc[base];
}
