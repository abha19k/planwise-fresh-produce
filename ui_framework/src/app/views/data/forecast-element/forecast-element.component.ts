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
import { debounceTime, distinctUntilChanged } from 'rxjs/operators';

interface IForecastElement {
  ProductID: string;
  ChannelID: string;
  LocationID: string;
  Level: string;
  IsActive: string;
}

interface SavedSearch { id?: number; name: string; query: string; created_at?: string; }

interface ForecastElementResponse {
  count: number;
  rows: any[];
}

@Component({
  standalone: true,
  selector: 'app-forecast-element',
  templateUrl: 'forecast-element.component.html',
  styleUrls: ['forecast-element.component.scss'],
  imports: [
    CommonModule, ReactiveFormsModule, HttpClientModule,
    TextColorDirective, CardComponent, CardBodyComponent, CardHeaderComponent, CardFooterComponent,
    RowComponent, ColComponent, ButtonDirective, IconDirective, TableDirective
  ]
})
export class ForecastElementComponent implements OnInit {
  private http: HttpClient = inject(HttpClient);
  private readonly API = 'http://127.0.0.1:8000/api';

  // Data
  public forecastElements: IForecastElement[] = [];
  public filteredUnits: IForecastElement[] = [];

  // Typed filters -> backend (/api/forecastelements)
  public productSearch = new FormControl('');
  public channelSearch = new FormControl('');
  public locationSearch = new FormControl('');
  public levelSearch = new FormControl('');      // typed "1" / "2" / "3" etc
  public isActiveSearch = new FormControl('');   // typed "true"/"false" or blank

  // Saved searches
  public savedSearches: SavedSearch[] = [];
  public selectedSavedIndex = new FormControl<number>(-1);

  // UI state
  public loading = false;
  public errorMessage: string | null = null;

  // Pagination & sorting
  public currentPage = 1;
  public itemsPerPage = 20;
  public totalPages = 1;
  public sortColumn: keyof IForecastElement | '' = '';
  public sortAsc = true;

  ngOnInit(): void {
    this.refreshSavedSearches();

    // show nothing by default
    this.resetResults();

    // Auto-search when typing (debounced)
    const watchers = [
      this.productSearch.valueChanges,
      this.channelSearch.valueChanges,
      this.locationSearch.valueChanges,
      this.levelSearch.valueChanges,
      this.isActiveSearch.valueChanges,
    ];

    watchers.forEach(obs => {
      obs.pipe(debounceTime(250), distinctUntilChanged())
        .subscribe(() => this.autoSearch());
    });
  }

  private resetResults() {
    this.forecastElements = [];
    this.filteredUnits = [];
    this.currentPage = 1;
    this.updatePagination();
  }

  private hasAnyInput(): boolean {
    const vals = [
      this.productSearch.value,
      this.channelSearch.value,
      this.locationSearch.value,
      this.levelSearch.value,
      this.isActiveSearch.value,
    ];
    return vals.some(v => (v || '').trim().length > 0);
  }

  // ----- Saved searches -----
  refreshSavedSearches() {
    this.http.get<SavedSearch[]>(`${this.API}/saved-searches`).subscribe({
      next: rows => { this.savedSearches = rows || []; },
      error: e => console.error('saved-searches failed', e)
    });
  }

  /**
   * Saved searches were originally created for /api/search (query language).
   * For Forecast Element page, we just "best-effort" parse out:
   * productid:..., channelid:..., locationid:...
   * If none found -> show a clear error.
   */
  async loadFromSaved() {
    this.errorMessage = null;
    const idx = this.selectedSavedIndex.value ?? -1;
    if (idx < 0 || idx >= this.savedSearches.length) {
      this.errorMessage = 'Please choose a saved search.';
      return;
    }

    const q = this.savedSearches[idx].query || '';
    const lower = q.toLowerCase();

    const getToken = (key: string) => {
      // matches key:value or key:"value with spaces"
      const m = new RegExp(`${key}:(".*?"|\\S+)`, 'i').exec(q);
      if (!m) return '';
      let v = (m[1] || '').trim();
      if (v.startsWith('"') && v.endsWith('"')) v = v.slice(1, -1);
      v = v.replace(/\*/g, ''); // user typed query wildcards -> typed search doesn't need them
      return v;
    };

    const p = getToken('productid');
    const c = getToken('channelid');
    const l = getToken('locationid');

    if (!p && !c && !l) {
      this.errorMessage = 'This saved search does not contain productid/channelid/locationid tokens. Use typed search instead.';
      return;
    }

    // set filters (this triggers autoSearch via valueChanges)
    this.productSearch.setValue(p, { emitEvent: true });
    this.channelSearch.setValue(c, { emitEvent: true });
    this.locationSearch.setValue(l, { emitEvent: true });
  }

  // ----- Auto search -----
  private async autoSearch() {
    this.errorMessage = null;

    if (!this.hasAnyInput()) {
      this.resetResults();
      return;
    }

    await this.fetchForecastElements();
  }

  async runTypedSearch() {
    this.errorMessage = null;
    if (!this.hasAnyInput()) {
      this.errorMessage = 'Enter at least one filter (Product/Channel/Location/Level/IsActive).';
      return;
    }
    await this.fetchForecastElements();
  }

  private async fetchForecastElements() {
    this.loading = true;
    try {
      const params = new HttpParams()
        .set('productid', (this.productSearch.value || '').trim())
        .set('channelid', (this.channelSearch.value || '').trim())
        .set('locationid', (this.locationSearch.value || '').trim())
        .set('level', (this.levelSearch.value || '').trim())
        .set('isactive', (this.isActiveSearch.value || '').trim())
        .set('limit', 20000)
        .set('offset', 0);

      const res = await firstValueFrom(
        this.http.get<ForecastElementResponse>(`${this.API}/forecastelements`, { params })
      );

      const rows = res?.rows ?? [];
      this.forecastElements = rows.map((r: any) => ({
        ProductID: String(r.ProductID ?? '').trim(),
        ChannelID: String(r.ChannelID ?? '').trim(),
        LocationID: String(r.LocationID ?? '').trim(),
        Level: String(r.Level ?? '').trim(),
        IsActive: String(r.IsActive ?? '').trim(),
      }));

      this.filteredUnits = [...this.forecastElements];
      this.currentPage = 1;
      this.updatePagination();
    } catch (e: any) {
      console.error('GET /forecastelements failed', e);
      this.errorMessage = e?.error?.detail || e?.message || 'Failed to load forecast elements.';
      this.resetResults();
    } finally {
      this.loading = false;
    }
  }

  // ----- Local table features -----
  clearAllFilters() {
    // clearing should also clear results (autoSearch handles it)
    this.productSearch.setValue('');
    this.channelSearch.setValue('');
    this.locationSearch.setValue('');
    this.levelSearch.setValue('');
    this.isActiveSearch.setValue('');
  }

  exportToCSV() {
    if (!this.filteredUnits.length) return;
    const header = ['ProductID', 'ChannelID', 'LocationID', 'Level', 'IsActive'];
    const rows = this.filteredUnits.map(row =>
      header.map(f => `"${String((row as any)[f] ?? '').replace(/"/g, '""')}"`).join(',')
    );
    const csvContent = [header.join(','), ...rows].join('\r\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.setAttribute('download', 'ForecastElements_Filtered.csv');
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  sortBy(column: keyof IForecastElement) {
    if (this.sortColumn === column) this.sortAsc = !this.sortAsc;
    else { this.sortColumn = column; this.sortAsc = true; }

    this.filteredUnits.sort((a, b) => {
      const va = (a[column] as any)?.toLowerCase?.() ?? String(a[column] ?? '');
      const vb = (b[column] as any)?.toLowerCase?.() ?? String(b[column] ?? '');
      return this.sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    });
  }

  updatePagination() {
    this.totalPages = Math.max(1, Math.ceil(this.filteredUnits.length / this.itemsPerPage));
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

  get paginatedData(): IForecastElement[] {
    const start = (this.currentPage - 1) * this.itemsPerPage;
    return this.filteredUnits.slice(start, start + this.itemsPerPage);
  }

  setPage(page: number) {
    const total = Math.max(1, this.totalPages);
    this.currentPage = Math.min(Math.max(1, page), total);
  }
}
