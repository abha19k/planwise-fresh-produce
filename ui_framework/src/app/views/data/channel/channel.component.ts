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

interface IChannel {
  ChannelID: string;
  ChannelDescr: string;
  Level: string;
}

interface KeyTriplet { ProductID: string; ChannelID: string; LocationID: string; }
interface SearchResult { query: string; count: number; keys: KeyTriplet[]; }
interface SavedSearch { id?: number; name: string; query: string; created_at?: string; }

@Component({
  standalone: true,
  selector: 'app-channel',
  templateUrl: 'channel.component.html',
  styleUrls: ['channel.component.scss'],
  imports: [
    CommonModule, ReactiveFormsModule, HttpClientModule,
    TextColorDirective, CardComponent, CardBodyComponent, CardHeaderComponent, CardFooterComponent,
    RowComponent, ColComponent, ButtonDirective, IconDirective, TableDirective
  ]
})
export class ChannelComponent implements OnInit {
  private http: HttpClient = inject(HttpClient);
  private readonly API = 'http://127.0.0.1:8000/api';

  public channelData: IChannel[] = [];
  public searchResults: IChannel[] = [];
  public filteredData: IChannel[] = [];

  public searchField = new FormControl<'ChannelID' | 'ChannelDescr' | 'Level'>('ChannelID');
  public searchTerm = new FormControl('');

  public savedSearches: SavedSearch[] = [];
  public selectedSavedIndex = new FormControl<number>(-1);

  public loading = false;
  public errorMessage: string | null = null;

  /** ✅ NEW: show results only after search/saved search */
  public hasRunSearch = false;

  public currentPage = 1;
  public itemsPerPage = 5;
  public totalPages = 1;
  public sortColumn: keyof IChannel | '' = '';
  public sortAsc: boolean = true;

  ngOnInit(): void {
    this.loadChannels();
    this.refreshSavedSearches();

    this.searchTerm.valueChanges.subscribe(term => this.performSearch(term || ''));
    this.searchField.valueChanges.subscribe(() => this.performSearch(this.searchTerm.value || ''));
  }

  private loadChannels() {
    this.loading = true;
    this.http.get<IChannel[]>(`${this.API}/channels`).subscribe({
      next: rows => {
        this.channelData = (rows || []).map(c => ({
          ChannelID: String(c.ChannelID ?? ''),
          ChannelDescr: String(c.ChannelDescr ?? ''),
          Level: String((c as any).Level ?? '')
        }));

        // ✅ no data shown by default
        this.searchResults = [];
        this.filteredData = [];
        this.hasRunSearch = false;
        this.currentPage = 1;
        this.updatePagination();
      },
      error: err => {
        console.error('GET /channels failed', err);
        this.errorMessage = 'Failed to load channels.';
      },
      complete: () => { this.loading = false; }
    });
  }

  refreshSavedSearches() {
    this.http.get<SavedSearch[]>(`${this.API}/saved-searches`).subscribe({
      next: rows => { this.savedSearches = rows || []; },
      error: e => { console.error('saved-searches failed', e); }
    });
  }

  private savedQueryHasChannelField(q: string): boolean {
    const s = (q || '').toLowerCase();
    return ['channelid:', 'channeldescr:', 'channellevel:'].some(tag => s.includes(tag));
  }

  async loadFromSaved() {
    this.errorMessage = null;
    const idx = this.selectedSavedIndex.value ?? -1;

    if (idx < 0 || idx >= this.savedSearches.length) {
      this.errorMessage = 'Please choose a saved search.';
      return;
    }

    const q = this.savedSearches[idx].query;

    if (!this.savedQueryHasChannelField(q)) {
      this.errorMessage = 'This saved search does not include channel attributes. Channel page only runs searches with channel fields.';
      this.searchResults = [];
      this.filteredData = [];
      this.hasRunSearch = true; // ✅ user attempted to load results
      this.updatePagination();
      return;
    }

    await this.runQueryAndFilterChannels(q);
  }

  private backendFieldFor(uiField: string): string | null {
    switch ((uiField || '').toLowerCase()) {
      case 'channelid': return 'channelid';
      case 'channeldescr': return 'channeldescr';
      case 'level': return 'channellevel';
      default: return null;
    }
  }

  private buildChannelQuery(fieldUI: string, term: string): string | null {
    const field = this.backendFieldFor(fieldUI);
    if (!field) return null;

    let value = (term || '').trim();
    if (!value) return null;

    if (!/[.*%]/.test(value)) value = `*${value}*`;
    if (/\s/.test(value)) value = `"${value}"`;

    return `${field}:${value}`;
  }

  async performSearch(term: string) {
    this.errorMessage = null;

    if (!term?.trim()) {
      // ✅ cleared: hide table again
      this.searchResults = [];
      this.filteredData = [];
      this.hasRunSearch = false;
      this.currentPage = 1;
      this.updatePagination();
      return;
    }

    const q = this.buildChannelQuery(this.searchField.value || 'ChannelID', term);
    if (!q) {
      this.errorMessage = 'Please choose a valid channel attribute.';
      this.searchResults = [];
      this.filteredData = [];
      this.hasRunSearch = true;
      this.updatePagination();
      return;
    }

    await this.runQueryAndFilterChannels(q);
  }

  private async runQueryAndFilterChannels(q: string) {
    this.loading = true;
    this.hasRunSearch = true;

    try {
      const params = new HttpParams().set('q', q).set('limit', 20000).set('offset', 0);
      const res = await firstValueFrom(this.http.get<SearchResult>(`${this.API}/search`, { params }));

      const keys = res?.keys ?? [];
      if (!keys.length) {
        this.searchResults = [];
        this.filteredData = [];
        this.currentPage = 1;
        this.updatePagination();
        return;
      }

      const allowed = new Set(keys.map(k => k.ChannelID));
      this.searchResults = this.channelData.filter(c => allowed.has(c.ChannelID));
      this.filteredData = [...this.searchResults];

      this.currentPage = 1;
      this.updatePagination();
    } catch (e: any) {
      console.error('Channel search failed', e);
      this.errorMessage = e?.error?.detail || e?.message || 'Search failed.';
      this.searchResults = [];
      this.filteredData = [];
      this.currentPage = 1;
      this.updatePagination();
    } finally {
      this.loading = false;
    }
  }

  clearSearch() {
    this.searchTerm.setValue('');
    // performSearch() will handle resetting hasRunSearch + arrays
  }

  exportToCSV() {
    const data = this.filteredData;
    if (!data.length) return;

    const header = Object.keys(data[0]);
    const rows = data.map(row => header.map(field => `"${(row as any)[field]}"`).join(','));
    const csvContent = [header.join(','), ...rows].join('\r\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'Channel_Search_Results.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  sortBy(column: keyof IChannel) {
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

  get paginatedData(): IChannel[] {
    const start = (this.currentPage - 1) * this.itemsPerPage;
    return this.filteredData.slice(start, start + this.itemsPerPage);
  }

  setPage(page: number) {
    const total = Math.max(1, this.totalPages);
    this.currentPage = Math.min(Math.max(1, page), total);
  }

  pageRange(): number[] {
    return Array.from({ length: this.totalPages }, (_, i) => i + 1);
  }
}
