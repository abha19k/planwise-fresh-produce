import { CommonModule } from '@angular/common';
import { Component, OnInit, inject } from '@angular/core';
import { FormControl, ReactiveFormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule, HttpParams } from '@angular/common/http';
import {
  ButtonDirective, CardBodyComponent, CardComponent, CardFooterComponent, CardHeaderComponent, ColComponent,
  RowComponent, TableDirective, TextColorDirective
} from '@coreui/angular';
import { IconDirective } from '@coreui/icons-angular';
import { firstValueFrom } from 'rxjs';

interface ILocation {
  LocationID: string;
  LocationDescr: string;
  Level: string;
  IsActive: string;
}

interface KeyTriplet { ProductID: string; ChannelID: string; LocationID: string; }
interface SearchResult { query: string; count: number; keys: KeyTriplet[]; }
interface SavedSearch { id?: number; name: string; query: string; created_at?: string; }

type LocationField = 'LocationID' | 'LocationDescr' | 'Level' | 'IsActive';

@Component({
  standalone: true,
  selector: 'app-location',
  templateUrl: 'location.component.html',
  styleUrls: ['location.component.scss'],
  imports: [
    CommonModule, ReactiveFormsModule, HttpClientModule,
    TextColorDirective, CardComponent, CardBodyComponent, CardHeaderComponent, CardFooterComponent,
    RowComponent, ColComponent, ButtonDirective, IconDirective, TableDirective
  ]
})
export class LocationComponent implements OnInit {
  private http: HttpClient = inject(HttpClient);
  private readonly API = 'http://127.0.0.1:8000/api';

  // Master data (loaded once)
  public locationData: ILocation[] = [];

  // Visible data (ONLY after search/load)
  public searchResults: ILocation[] = [];
  public filteredData: ILocation[] = [];

  // Typed search (local)
  public searchField = new FormControl<LocationField>('LocationID');
  public searchTerm = new FormControl('');

  // Saved searches
  public savedSearches: SavedSearch[] = [];
  public selectedSavedIndex = new FormControl<number>(-1);

  // UI state
  public loading = false;
  public errorMessage: string | null = null;

  // Pagination & sorting (over filteredData)
  public currentPage = 1;
  public itemsPerPage = 5;
  public totalPages = 1;
  public sortColumn: keyof ILocation | '' = '';
  public sortAsc: boolean = true;

  ngOnInit(): void {
    this.loadLocations();
    this.refreshSavedSearches();

    // local typed search
    this.searchTerm.valueChanges.subscribe(term => this.performLocalSearch(term || ''));
    this.searchField.valueChanges.subscribe(() => this.performLocalSearch(this.searchTerm.value || ''));
  }

  /** Load location master from backend once */
  private loadLocations() {
    this.loading = true;
    this.http.get<ILocation[]>(`${this.API}/locations`).subscribe({
      next: rows => {
        this.locationData = (rows || []).map(l => ({
          LocationID: String((l as any).LocationID ?? '').trim(),
          LocationDescr: String((l as any).LocationDescr ?? '').trim(),
          Level: String((l as any).Level ?? '').trim(),
          IsActive: String((l as any).IsActive ?? '').trim(),
        }));

        // IMPORTANT: show nothing by default
        this.clearVisible();
      },
      error: err => {
        console.error('GET /locations failed', err);
        this.errorMessage = 'Failed to load locations.';
        this.clearVisible();
      },
      complete: () => { this.loading = false; }
    });
  }

  private clearVisible() {
    this.searchResults = [];
    this.filteredData = [];
    this.currentPage = 1;
    this.updatePagination();
  }

  /** Saved searches list */
  refreshSavedSearches() {
    this.http.get<SavedSearch[]>(`${this.API}/saved-searches`).subscribe({
      next: rows => { this.savedSearches = rows || []; },
      error: e => { console.error('saved-searches failed', e); }
    });
  }

  /** Saved search must contain location fields to be eligible for this page */
  private savedQueryHasLocationField(q: string): boolean {
    const s = (q || '').toLowerCase();
    return ['locationid:', 'locationdescr:', 'locationlevel:', 'isactive:'].some(tag => s.includes(tag));
  }

  /** Does query contain any non-location fields (product/channel constraints)? */
  private queryHasNonLocationField(q: string): boolean {
    const s = (q || '').toLowerCase();
    // crude but effective: if it contains product*/channel* fields, it’s not location-only
    return ['product', 'channel'].some(k => s.includes(`${k}id:`) || s.includes(`${k}descr:`) || s.includes(`${k}level:`));
  }

  async loadFromSaved() {
    this.errorMessage = null;

    const idx = this.selectedSavedIndex.value ?? -1;
    if (idx < 0 || idx >= this.savedSearches.length) {
      this.errorMessage = 'Please choose a saved search.';
      return;
    }

    const q = this.savedSearches[idx].query || '';
    if (!this.savedQueryHasLocationField(q)) {
      this.errorMessage = 'This saved search does not include location attributes. Location page only runs searches with location fields.';
      this.clearVisible();
      return;
    }

    // If the query only talks about location fields → run locally on master table (Option A)
    if (!this.queryHasNonLocationField(q)) {
      this.runLocalQueryString(q);
      return;
    }

    // Otherwise: use /api/search to respect product/channel constraints (feasible locations only)
    await this.runTripletQueryAndFilterLocations(q);
  }

  // -------------------------
  // Local search (Option A)
  // -------------------------
  private matchWithWildcards(value: string, term: string): boolean {
    const v = (value ?? '').toLowerCase();
    const t = (term ?? '').trim().toLowerCase();
    if (!t) return true;

    // If user typed no wildcard, treat as "contains"
    if (!t.includes('*')) return v.includes(t);

    // Convert '*' wildcard to regex
    const escaped = t.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*');
    const re = new RegExp(`^${escaped}$`);
    return re.test(v);
  }

  private performLocalSearch(term: string) {
    this.errorMessage = null;

    if (!term?.trim()) {
      // show nothing until user searches
      this.clearVisible();
      return;
    }

    const field = (this.searchField.value || 'LocationID') as LocationField;
    const t = term.trim();

    this.searchResults = this.locationData.filter(row =>
      this.matchWithWildcards(String((row as any)[field] ?? ''), t)
    );

    this.filteredData = [...this.searchResults];
    this.currentPage = 1;
    this.updatePagination();
  }

  /** Parse query like: locationlevel:*2* locationid:*NL* and apply locally */
  private runLocalQueryString(q: string) {
    // Support the same "field:value" tokens as backend.
    // We only apply location fields locally here.
    const tokens = Array.from(q.matchAll(/(\w+):(".*?"|\S+)/g)).map(m => ({
      field: (m[1] || '').toLowerCase(),
      value: (m[2] || '').trim()
    }));

    const getVal = (raw: string) => {
      let v = raw;
      if (v.startsWith('"') && v.endsWith('"')) v = v.slice(1, -1);
      return v;
    };

    const predicates: Array<(row: ILocation) => boolean> = [];

    for (const t of tokens) {
      const val = getVal(t.value);

      if (t.field === 'locationid') {
        predicates.push(r => this.matchWithWildcards(r.LocationID, val));
      } else if (t.field === 'locationdescr') {
        predicates.push(r => this.matchWithWildcards(r.LocationDescr, val));
      } else if (t.field === 'locationlevel') {
        predicates.push(r => this.matchWithWildcards(r.Level, val));
      } else if (t.field === 'isactive') {
        predicates.push(r => this.matchWithWildcards(r.IsActive, val));
      }
    }

    const out = this.locationData.filter(row => predicates.every(p => p(row)));
    this.searchResults = out;
    this.filteredData = [...out];
    this.currentPage = 1;
    this.updatePagination();
  }

  // -------------------------
  // Triplet search (only when needed)
  // -------------------------
  private async runTripletQueryAndFilterLocations(q: string) {
    this.loading = true;
    try {
      const params = new HttpParams().set('q', q).set('limit', 20000).set('offset', 0);
      const res = await firstValueFrom(this.http.get<SearchResult>(`${this.API}/search`, { params }));

      const keys = res?.keys ?? [];
      if (!keys.length) {
        this.clearVisible();
        return;
      }

      const allowed = new Set(keys.map(k => k.LocationID));
      this.searchResults = this.locationData.filter(l => allowed.has(l.LocationID));
      this.filteredData = [...this.searchResults];

      this.currentPage = 1;
      this.updatePagination();
    } catch (e: any) {
      console.error('Location triplet search failed', e);
      this.errorMessage = e?.error?.detail || e?.message || 'Search failed.';
      this.clearVisible();
    } finally {
      this.loading = false;
    }
  }

  // -------------------------
  // Export / Sort / Paging
  // -------------------------
  exportToCSV() {
    const data = this.filteredData;
    if (!data.length) return;

    const header = Object.keys(data[0] as any);
    const rows = data.map(row =>
      header.map(field => {
        const cell = (row as any)[field] ?? '';
        const escaped = String(cell).replace(/"/g, '""');
        return `"${escaped}"`;
      }).join(',')
    );

    const csv = [header.join(','), ...rows].join('\r\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'Location_Search_Results.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  sortBy(column: keyof ILocation) {
    if (this.sortColumn === column) this.sortAsc = !this.sortAsc;
    else { this.sortColumn = column; this.sortAsc = true; }

    this.filteredData.sort((a, b) => {
      const valA = (a[column] as any)?.toLowerCase?.() ?? String(a[column] ?? '');
      const valB = (b[column] as any)?.toLowerCase?.() ?? String(b[column] ?? '');
      return this.sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
    });
  }

  updatePagination() {
    this.totalPages = Math.max(1, Math.ceil(this.filteredData.length / this.itemsPerPage));
    if (this.currentPage > this.totalPages) this.currentPage = this.totalPages;
  }

  visiblePages(): number[] {
    const total = this.totalPages || 1;
    const current = Math.min(Math.max(this.currentPage, 1), total);
    const windowSize = 5;

    if (total <= windowSize) return Array.from({ length: total }, (_, i) => i + 1);

    let start = current - Math.floor(windowSize / 2);
    let end = current + Math.floor(windowSize / 2);

    if (start < 1) { start = 1; end = windowSize; }
    if (end > total) { end = total; start = total - windowSize + 1; }

    return Array.from({ length: end - start + 1 }, (_, i) => start + i);
  }

  get paginatedData(): ILocation[] {
    const start = (this.currentPage - 1) * this.itemsPerPage;
    return this.filteredData.slice(start, start + this.itemsPerPage);
  }

  setPage(page: number) {
    const total = Math.max(1, this.totalPages);
    this.currentPage = Math.min(Math.max(1, page), total);
  }
}
