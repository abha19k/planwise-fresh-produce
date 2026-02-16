import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit, inject } from '@angular/core';
import { FormControl, ReactiveFormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule, HttpParams } from '@angular/common/http';
import {
  ButtonDirective, CardBodyComponent, CardComponent, CardFooterComponent, CardHeaderComponent, ColComponent,
  RowComponent, TableDirective, TextColorDirective
} from '@coreui/angular';
import { IconDirective } from '@coreui/icons-angular';
import { Subject, combineLatest, of } from 'rxjs';
import { catchError, debounceTime, distinctUntilChanged, finalize, map, startWith, switchMap, takeUntil, tap } from 'rxjs/operators';

interface IProduct {
  ProductID: string;
  ProductDescr: string;
  Level: string;
  BusinessUnit: string;
  IsDailyForecastRequired: string;
  IsNew: string;
  ProductFamily: string;
}

interface KeyTriplet { ProductID: string; ChannelID: string; LocationID: string; }
interface SearchResult { query: string; count: number; keys: KeyTriplet[]; }
interface SavedSearch { id?: number; name: string; query: string; created_at?: string; }

type ProductField =
  | 'ProductID'
  | 'ProductDescr'
  | 'Level'
  | 'BusinessUnit'
  | 'IsDailyForecastRequired'
  | 'IsNew'
  | 'ProductFamily';

@Component({
  standalone: true,
  selector: 'app-product',
  templateUrl: 'product.component.html',
  styleUrls: ['product.component.scss'],
  imports: [
    CommonModule, ReactiveFormsModule, HttpClientModule,
    TextColorDirective, CardComponent, CardBodyComponent, CardHeaderComponent, CardFooterComponent,
    RowComponent, ColComponent, ButtonDirective, IconDirective, TableDirective
  ]
})
export class ProductComponent implements OnInit, OnDestroy {
  private http: HttpClient = inject(HttpClient);
  private destroy$ = new Subject<void>();

  private readonly API = 'http://127.0.0.1:8000/api';

  /** Loaded once */
  public productData: IProduct[] = [];

  /** Derived views */
  public searchResults: IProduct[] = [];
  public filteredData: IProduct[] = [];

  /** Search inputs */
  public searchField = new FormControl<ProductField>('ProductID', { nonNullable: true });
  public searchTerm = new FormControl<string>('', { nonNullable: true });

  /** Saved searches */
  public savedSearches: SavedSearch[] = [];
  public selectedSavedIndex = new FormControl<number>(-1, { nonNullable: true });

  /** Dropdown filters */
  public selectedBusinessUnit = new FormControl<string>('', { nonNullable: true });
  public selectedIsNew = new FormControl<string>('', { nonNullable: true });
  public businessUnitList: string[] = [];
  public isNewList: string[] = [];

  /** UI state */
  public loading = false;
  public errorMessage: string | null = null;

  /** Pagination & sorting */
  public currentPage = 1;
  public itemsPerPage = 5;
  public totalPages = 1;
  public sortColumn: keyof IProduct | '' = '';
  public sortAsc = true;

  /** ✅ Controls whether table should show anything */
  public hasActiveFilter = false;

  ngOnInit(): void {
    this.loadProducts();
    this.refreshSavedSearches();

    const searchField$ = this.searchField.valueChanges.pipe(startWith(this.searchField.value));
    const searchTerm$ = this.searchTerm.valueChanges.pipe(
      startWith(this.searchTerm.value),
      debounceTime(250),
      distinctUntilChanged()
    );

    const bu$ = this.selectedBusinessUnit.valueChanges.pipe(startWith(this.selectedBusinessUnit.value));
    const isNew$ = this.selectedIsNew.valueChanges.pipe(startWith(this.selectedIsNew.value));

    // Any change in (search or dropdown) recomputes result set
    combineLatest([searchField$, searchTerm$, bu$, isNew$])
      .pipe(
        takeUntil(this.destroy$),
        tap(() => { this.errorMessage = null; }),
        switchMap(([field, term, bu, isNew]) => {
          const t = (term || '').trim();
          const buv = (bu || '').trim();
          const inv = (isNew || '').trim();

          // ✅ show results only if search term exists OR dropdown filter chosen
          this.hasActiveFilter = !!t || !!buv || !!inv;

          if (!this.hasActiveFilter) {
            // nothing selected -> show nothing
            return of({ mode: 'idle' as const, products: [] as IProduct[] });
          }

          // If dropdown only (no search term): base set = all products, then filter client-side
          if (!t) {
            const base = [...this.productData];
            return of({ mode: 'dropdownOnly' as const, products: base });
          }

          // Otherwise: run backend key-search, then filter products
          const q = this.buildProductQuery(field, t);
          if (!q) {
            return of({ mode: 'error' as const, products: [] as IProduct[], msg: 'Please choose a valid product attribute.' });
          }

          this.loading = true;
          const params = new HttpParams().set('q', q).set('limit', 20000).set('offset', 0);

          return this.http.get<SearchResult>(`${this.API}/search`, { params }).pipe(
            map(res => {
              const keys = res?.keys ?? [];
              if (!keys.length) return { mode: 'empty' as const, products: [] as IProduct[] };
              const allowed = new Set(keys.map(k => String(k.ProductID)));
              const products = this.productData.filter(p => allowed.has(p.ProductID));
              return { mode: 'ok' as const, products };
            }),
            catchError((e: any) => {
              const msg = e?.error?.detail || e?.message || 'Search failed.';
              return of({ mode: 'error' as const, products: [] as IProduct[], msg });
            }),
            finalize(() => { this.loading = false; })
          );
        })
      )
      .subscribe(result => {
        if (result.mode === 'error') this.errorMessage = (result as any).msg || 'Search failed.';
        this.searchResults = result.products;
        this.applyDropdownFilters(); // applies BU/IsNew + updates pagination
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  /** Load products once (still needed for dropdown values + filtering after key-search) */
  private loadProducts() {
    this.loading = true;
    this.http.get<IProduct[]>(`${this.API}/products`).pipe(
      finalize(() => { this.loading = false; }),
      takeUntil(this.destroy$)
    ).subscribe({
      next: rows => {
        this.productData = (rows || []).map(p => ({
          ProductID: String((p as any).ProductID ?? ''),
          ProductDescr: String((p as any).ProductDescr ?? ''),
          Level: String((p as any).Level ?? ''),
          BusinessUnit: String((p as any).BusinessUnit ?? ''),
          IsDailyForecastRequired: String((p as any).IsDailyForecastRequired ?? ''),
          IsNew: String((p as any).IsNew ?? ''),
          ProductFamily: String((p as any).ProductFamily ?? '')
        }));

        this.businessUnitList = [...new Set(this.productData.map(p => p.BusinessUnit).filter(Boolean))].sort();
        this.isNewList = [...new Set(this.productData.map(p => p.IsNew).filter(Boolean))].sort();

        // ✅ default: show nothing
        this.searchResults = [];
        this.filteredData = [];
        this.updatePagination();
      },
      error: err => {
        console.error('GET /products failed', err);
        this.errorMessage = 'Failed to load products.';
        this.productData = [];
        this.searchResults = [];
        this.filteredData = [];
        this.updatePagination();
      }
    });
  }

  /** Saved searches */
  refreshSavedSearches() {
    this.http.get<SavedSearch[]>(`${this.API}/saved-searches`)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: rows => { this.savedSearches = rows || []; },
        error: e => { console.error('saved-searches failed', e); }
      });
  }

  private savedQueryHasProductField(q: string): boolean {
    const s = (q || '').toLowerCase();
    return [
      'productid:', 'productdescr:', 'businessunit:',
      'isdailyforecastrequired:', 'isnew:', 'productfamily:',
      'productlevel:'
    ].some(tag => s.includes(tag));
  }

  async loadFromSaved() {
    this.errorMessage = null;
    const idx = this.selectedSavedIndex.value ?? -1;

    if (idx < 0 || idx >= this.savedSearches.length) {
      this.errorMessage = 'Please choose a saved search.';
      return;
    }

    const q = this.savedSearches[idx].query;
    if (!this.savedQueryHasProductField(q)) {
      this.errorMessage = 'This saved search does not include product attributes. Product page only runs searches with product fields.';
      this.hasActiveFilter = false;
      this.searchResults = [];
      this.filteredData = [];
      this.updatePagination();
      return;
    }

    this.loading = true;
    this.hasActiveFilter = true; // ✅ show results
    try {
      const params = new HttpParams().set('q', q).set('limit', 20000).set('offset', 0);
      const res = await this.http.get<SearchResult>(`${this.API}/search`, { params }).toPromise();
      const keys = res?.keys ?? [];
      const allowed = new Set(keys.map(k => String(k.ProductID)));
      this.searchResults = this.productData.filter(p => allowed.has(p.ProductID));

      // keep dropdown filters applied on top
      this.applyDropdownFilters();
    } catch (e: any) {
      console.error('Saved search failed', e);
      this.errorMessage = e?.error?.detail || e?.message || 'Saved search failed.';
      this.searchResults = [];
      this.applyDropdownFilters();
    } finally {
      this.loading = false;
    }
  }

  private backendFieldFor(uiField: ProductField): string | null {
    switch ((uiField || '').toLowerCase()) {
      case 'productid': return 'productid';
      case 'productdescr': return 'productdescr';
      case 'businessunit': return 'businessunit';
      case 'isdailyforecastrequired': return 'isdailyforecastrequired';
      case 'isnew': return 'isnew';
      case 'productfamily': return 'productfamily';
      case 'level': return 'productlevel';
      default: return null;
    }
  }

  private buildProductQuery(fieldUI: ProductField, term: string): string | null {
    const field = this.backendFieldFor(fieldUI);
    if (!field) return null;

    let value = (term || '').trim();
    if (!value) return null;

    if (!/[.*%]/.test(value)) value = `*${value}*`;
    if (/\s/.test(value)) value = `"${value}"`;

    return `${field}:${value}`;
  }

  /** Apply BU/IsNew filters on top of searchResults (or dropdown-only base) */
  applyDropdownFilters() {
    const bu = (this.selectedBusinessUnit.value || '').trim();
    const isNew = (this.selectedIsNew.value || '').trim();

    // ✅ if no active filter at all, show nothing
    if (!this.hasActiveFilter) {
      this.filteredData = [];
      this.currentPage = 1;
      this.updatePagination();
      return;
    }

    const base = this.searchResults || [];
    this.filteredData = base.filter(p =>
      (!bu || p.BusinessUnit === bu) &&
      (!isNew || p.IsNew === isNew)
    );

    this.currentPage = 1;
    this.updatePagination();
  }

  exportToCSV() {
    const data = this.filteredData;
    if (!data.length) return;

    const header = Object.keys(data[0]);
    const rows = data.map(row => header.map(field => `"${String((row as any)[field] ?? '').replace(/"/g, '""')}"`).join(','));
    const csvContent = [header.join(','), ...rows].join('\r\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'Product_Search_Results.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  sortBy(column: keyof IProduct) {
    if (this.sortColumn === column) this.sortAsc = !this.sortAsc;
    else { this.sortColumn = column; this.sortAsc = true; }

    const asc = this.sortAsc;
    this.filteredData = [...this.filteredData].sort((a, b) => {
      const valA = String((a[column] ?? '')).toLowerCase();
      const valB = String((b[column] ?? '')).toLowerCase();
      return asc ? valA.localeCompare(valB) : valB.localeCompare(valA);
    });

    this.currentPage = 1;
    this.updatePagination();
  }

  updatePagination() {
    this.totalPages = Math.max(1, Math.ceil(this.filteredData.length / this.itemsPerPage));
    if (this.currentPage > this.totalPages) this.currentPage = this.totalPages;
  }

  get paginatedData(): IProduct[] {
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

  clearAllFilters() {
    this.searchTerm.setValue('');
    this.selectedBusinessUnit.setValue('');
    this.selectedIsNew.setValue('');
    this.selectedSavedIndex.setValue(-1);

    this.hasActiveFilter = false;
    this.searchResults = [];
    this.filteredData = [];
    this.updatePagination();
  }
}
