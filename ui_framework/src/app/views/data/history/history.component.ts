import { CommonModule } from '@angular/common';
import { Component, OnInit, inject } from '@angular/core';
import { FormControl, ReactiveFormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule, HttpParams } from '@angular/common/http';
import {
  ButtonDirective,
  CardBodyComponent,
  CardComponent,
  CardFooterComponent,
  CardHeaderComponent,
  ColComponent,
  RowComponent,
  TableDirective,
  TextColorDirective
} from '@coreui/angular';
import { IconDirective } from '@coreui/icons-angular';
import { firstValueFrom } from 'rxjs';

interface SavedSearch { id?: number; name: string; query: string; created_at?: string; }

interface IHistoryRow {
  Level: string;
  ProductID: string;
  ChannelID: string;
  LocationID: string;
  StartDate: string;
  EndDate: string;
  Period: string;
  Qty: number | string;
  NetPrice: number | string;
  ListPrice: number | string;
  Type: string;
}

interface HistorySearchResponse {
  field: string;
  term: string;
  count: number;
  rows: IHistoryRow[];
}

type HistoryTypeFilter = 'Both' | 'Normal-History' | 'Cleansed-History';

@Component({
  standalone: true,
  selector: 'app-history',
  templateUrl: 'history.component.html',
  styleUrls: ['history.component.scss'],
  imports: [
    CommonModule, ReactiveFormsModule, HttpClientModule,
    TextColorDirective, CardComponent, CardBodyComponent, CardHeaderComponent, CardFooterComponent,
    RowComponent, ColComponent, ButtonDirective, IconDirective, TableDirective
  ]
})
export class HistoryComponent implements OnInit {
  private http: HttpClient = inject(HttpClient);
  private readonly API = 'http://127.0.0.1:8000/api';

  // Filters
  public period = new FormControl<'Daily' | 'Weekly' | 'Monthly'>('Weekly');
  public level = new FormControl<string>(''); // optional: '' means all
  public searchField = new FormControl<'ProductID' | 'ChannelID' | 'LocationID'>('ProductID');
  public searchTerm = new FormControl<string>('');

  // ✅ NEW: Type filter (default BOTH so you see Normal + Cleansed together)
  public typeFilter = new FormControl<HistoryTypeFilter>('Both');

  // Saved searches
  public savedSearches: SavedSearch[] = [];
  public selectedSavedIndex = new FormControl<number>(-1);

  // Results
  public rows: IHistoryRow[] = [];
  public totalRows = 0;

  // UI state
  public loading = false;
  public errorMessage: string | null = null;

  // Pagination (server-side)
  public currentPage = 1;
  public pageSize = 200;

  // Mode tracking (typed vs saved query)
  private activeMode: 'typed' | 'saved' | null = null;
  private activeSavedQuery: string = '';

  ngOnInit(): void {
    this.refreshSavedSearches();
    this.clearRows();

    // Typed search subscriptions (only run when we're in typed mode)
    this.searchTerm.valueChanges.subscribe(() => {
      if (this.activeMode !== 'typed') return;
      this.currentPage = 1;
      this.runTypedSearch().catch(() => {});
    });

    this.searchField.valueChanges.subscribe(() => {
      if (this.activeMode !== 'typed') return;
      this.currentPage = 1;
      this.runTypedSearch().catch(() => {});
    });

    this.period.valueChanges.subscribe(() => {
      this.currentPage = 1;
      this.runActiveSearch().catch(() => {});
    });

    this.level.valueChanges.subscribe(() => {
      this.currentPage = 1;
      this.runActiveSearch().catch(() => {});
    });

    // ✅ NEW: Type filter changes should rerun search
    this.typeFilter.valueChanges.subscribe(() => {
      this.currentPage = 1;
      this.runActiveSearch().catch(() => {});
    });
  }

  get totalPages(): number {
    return Math.max(1, Math.ceil(this.totalRows / this.pageSize));
  }

  private clearRows() {
    this.rows = [];
    this.totalRows = 0;
    this.currentPage = 1;
  }

  refreshSavedSearches() {
    this.http.get<SavedSearch[]>(`${this.API}/saved-searches`).subscribe({
      next: rows => { this.savedSearches = rows || []; },
      error: e => { console.error('saved-searches failed', e); }
    });
  }

  /**
   * Start typed search mode (user types in search box).
   * Call this from UI interactions if you want to explicitly switch.
   * Here we auto-switch when user types (see key handler below).
   */
  public startTypedMode() {
    this.activeMode = 'typed';
    this.activeSavedQuery = '';
    // keep saved dropdown selection but it won't affect running
  }

  /** Optional: load saved search and RUN it against History (query language). */
  async useSavedQuery() {
    this.errorMessage = null;

    const idx = this.selectedSavedIndex.value ?? -1;
    if (idx < 0 || idx >= this.savedSearches.length) {
      this.errorMessage = 'Please choose a saved search.';
      return;
    }

    const q = (this.savedSearches[idx].query || '').trim();
    if (!q) {
      this.errorMessage = 'Saved query is empty.';
      return;
    }

    // switch mode
    this.activeMode = 'saved';
    this.activeSavedQuery = q;

    // show it in the input box for transparency, but DO NOT trigger typed search
    this.searchTerm.setValue('', { emitEvent: false }); // keep typed box clean

    this.currentPage = 1;
    await this.runSavedQuery(q);
  }

  /** Called by pagination and period/level/type changes */
  private async runActiveSearch() {
    if (this.activeMode === 'saved' && this.activeSavedQuery) {
      await this.runSavedQuery(this.activeSavedQuery);
      return;
    }
    // default to typed
    await this.runTypedSearch();
  }

  /** Map UI field to backend field (typed endpoint expects ProductID/ChannelID/LocationID) */
  private backendFieldFor(uiField: string): string {
    switch ((uiField || '').toLowerCase()) {
      case 'productid': return 'ProductID';
      case 'channelid': return 'ChannelID';
      case 'locationid': return 'LocationID';
      default: return 'ProductID';
    }
  }

  /** ✅ NEW: maps UI Type filter to backend parameter (omit when Both) */
  private backendTypeParam(): string | null {
    const t = this.typeFilter.value || 'Both';
    if (t === 'Both') return null;
    return t; // 'Normal-History' | 'Cleansed-History'
  }

  /** Typed search: uses /api/history/search */
  private async runTypedSearch() {
    this.errorMessage = null;

    const term = (this.searchTerm.value || '').trim();

    // If user cleared the box, reset
    if (!term) {
      this.clearRows();
      this.activeMode = 'typed';
      this.activeSavedQuery = '';
      return;
    }

    // If user edits the box while saved mode was active, switch to typed mode automatically
    if (this.activeMode !== 'typed') {
      this.activeMode = 'typed';
      this.activeSavedQuery = '';
    }

    const field = this.backendFieldFor(this.searchField.value || 'ProductID');
    const per = this.period.value || 'Weekly';
    const lvl = (this.level.value || '').trim();
    const offset = (this.currentPage - 1) * this.pageSize;
    const typeParam = this.backendTypeParam();

    this.loading = true;
    try {
      let params = new HttpParams()
        .set('field', field)
        .set('term', term)
        .set('period', per)
        .set('limit', String(this.pageSize))
        .set('offset', String(offset));

      if (lvl) params = params.set('level', lvl);
      if (typeParam) params = params.set('type', typeParam); // ✅ backend should filter on Type if provided

      const res = await firstValueFrom(
        this.http.get<HistorySearchResponse>(`${this.API}/history/search`, { params })
      );

      this.totalRows = Number(res?.count ?? 0);
      this.rows = (res?.rows ?? []).map((r: any) => ({
        Level: String(r.Level ?? ''),
        ProductID: String(r.ProductID ?? ''),
        ChannelID: String(r.ChannelID ?? ''),
        LocationID: String(r.LocationID ?? ''),
        StartDate: String(r.StartDate ?? ''),
        EndDate: String(r.EndDate ?? ''),
        Period: String(r.Period ?? ''),
        Qty: r.Qty ?? '',
        NetPrice: r.NetPrice ?? '',
        ListPrice: r.ListPrice ?? '',
        Type: String(r.Type ?? 'Normal-History'),
      }));
    } catch (e: any) {
      console.error('History typed search failed', e);
      this.errorMessage = e?.error?.detail || e?.message || 'History search failed.';
      this.clearRows();
    } finally {
      this.loading = false;
    }
  }

  /** Saved query mode: uses /api/history/by-query */
  private async runSavedQuery(q: string) {
    this.errorMessage = null;

    const per = this.period.value || 'Weekly';
    const lvl = (this.level.value || '').trim();
    const offset = (this.currentPage - 1) * this.pageSize;
    const typeParam = this.backendTypeParam();

    this.loading = true;
    try {
      let params = new HttpParams()
        .set('q', q)
        .set('limit', String(this.pageSize))
        .set('offset', String(offset));

      if (per) params = params.set('period', per);
      if (lvl) params = params.set('level', lvl);
      if (typeParam) params = params.set('type', typeParam); // ✅ backend should filter on Type if provided

      const res = await firstValueFrom(
        this.http.get<any>(`${this.API}/history/by-query`, { params })
      );

      this.totalRows = Number(res?.count ?? 0);
      this.rows = (res?.rows ?? []).map((r: any) => ({
        Level: String(r.Level ?? ''),
        ProductID: String(r.ProductID ?? ''),
        ChannelID: String(r.ChannelID ?? ''),
        LocationID: String(r.LocationID ?? ''),
        StartDate: String(r.StartDate ?? ''),
        EndDate: String(r.EndDate ?? ''),
        Period: String(r.Period ?? ''),
        Qty: r.Qty ?? '',
        NetPrice: r.NetPrice ?? '',
        ListPrice: r.ListPrice ?? '',
        Type: String(r.Type ?? 'Normal-History'),
      }));
    } catch (e: any) {
      console.error('History by-query failed', e);
      this.errorMessage = e?.error?.detail || e?.message || 'Saved search failed for history.';
      this.clearRows();
    } finally {
      this.loading = false;
    }
  }

  setPage(p: number) {
    const total = this.totalPages;
    const next = Math.min(Math.max(1, p), total);
    if (next === this.currentPage) return;
    this.currentPage = next;
    this.runActiveSearch().catch(() => {});
  }

  visiblePages(): number[] {
    const total = this.totalPages;
    const current = this.currentPage;
    const windowSize = 5;

    if (total <= windowSize) return Array.from({ length: total }, (_, i) => i + 1);

    let start = current - Math.floor(windowSize / 2);
    let end = current + Math.floor(windowSize / 2);

    if (start < 1) { start = 1; end = windowSize; }
    if (end > total) { end = total; start = total - windowSize + 1; }

    return Array.from({ length: end - start + 1 }, (_, i) => start + i);
  }

  clearAll() {
    this.searchTerm.setValue('', { emitEvent: false });
    this.level.setValue('', { emitEvent: false });
    this.searchField.setValue('ProductID', { emitEvent: false });
    this.period.setValue('Weekly', { emitEvent: false });
    this.typeFilter.setValue('Both', { emitEvent: false }); // ✅ NEW

    this.selectedSavedIndex.setValue(-1, { emitEvent: false });

    this.activeMode = null;
    this.activeSavedQuery = '';

    this.clearRows();
    this.errorMessage = null;
  }

  exportToCSV() {
    if (!this.rows.length) return;

    // ✅ include Type in export
    const header = ['Level','ProductID','ChannelID','LocationID','StartDate','EndDate','Period','Qty','NetPrice','ListPrice','Type'];
    const body = this.rows.map(r => header.map(h => `"${String((r as any)[h] ?? '').replace(/"/g, '""')}"`).join(','));
    const csv = [header.join(','), ...body].join('\r\n');

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;

    const modeTag = this.activeMode === 'saved' ? 'SavedQuery' : 'Typed';
    const termTag = (this.searchTerm.value || '').trim() || 'none';
    const typeTag = this.typeFilter.value || 'Both';

    a.download = `History_${modeTag}_${typeTag}_${this.period.value || 'Weekly'}_${this.searchField.value}_${termTag}_page${this.currentPage}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
}
