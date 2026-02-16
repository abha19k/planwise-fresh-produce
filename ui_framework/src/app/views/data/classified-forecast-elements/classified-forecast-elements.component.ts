import { CommonModule } from '@angular/common';
import { Component, OnInit, inject } from '@angular/core';
import { FormControl, ReactiveFormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule, HttpParams } from '@angular/common/http';
import { Router } from '@angular/router';
import {
  CardComponent, CardHeaderComponent, CardBodyComponent, CardFooterComponent,
  RowComponent, ColComponent, ButtonDirective, TextColorDirective, TableDirective
} from '@coreui/angular';
import { IconDirective } from '@coreui/icons-angular';

type PeriodView = 'Daily' | 'Weekly' | 'Monthly';
type PeriodSlug = 'daily' | 'weekly' | 'monthly';

interface RowVM {
  ProductID: string;
  ChannelID: string;
  LocationID: string;
  Period: PeriodView;
  adi: number | null;
  cv2: number | null;
  category: string;
  algorithm: string;
  isActive?: boolean;
  created_at?: string; // ISO timestamp
}

@Component({
  standalone: true,
  selector: 'app-classified-forecast-elements',
  templateUrl: './classified-forecast-elements.component.html',
  styleUrls: ['./classified-forecast-elements.component.scss'],
  imports: [
    CommonModule, ReactiveFormsModule, HttpClientModule,
    TextColorDirective, TableDirective,
    CardComponent, CardHeaderComponent, CardBodyComponent, CardFooterComponent,
    RowComponent, ColComponent, ButtonDirective, IconDirective
  ]
})
export class ClassifiedForecastElementsComponent implements OnInit {
  private http = inject(HttpClient);
  private router = inject(Router);

  readonly API = 'http://127.0.0.1:8000/api';

  period = new FormControl<PeriodView>('Daily', { nonNullable: true });
  includeInactive = new FormControl<boolean>(false, { nonNullable: true });

  rows: RowVM[] = [];
  loading = false;
  errorMsg: string | null = null;

  // sorting
  sortColumn: keyof RowVM | '' = '';
  sortAsc = true;

  ngOnInit(): void {
    this.loadRows();

    this.period.valueChanges.subscribe(() => this.loadRows());
    this.includeInactive.valueChanges.subscribe(() => this.loadRows());
  }

  private slug(): PeriodSlug {
    const p = this.period.value;
    return p === 'Weekly' ? 'weekly' : p === 'Monthly' ? 'monthly' : 'daily';
  }

  private toVM(r: any): RowVM {
    const slug = this.slug();
    const Period: PeriodView =
      (r.Period as PeriodView) ??
      (slug === 'weekly' ? 'Weekly' : slug === 'monthly' ? 'Monthly' : 'Daily');

    return {
      ProductID: String(r.ProductID ?? ''),
      ChannelID: String(r.ChannelID ?? ''),
      LocationID: String(r.LocationID ?? ''),
      Period,
      adi: r.ADI ?? r.adi ?? null,
      cv2: r.CV2 ?? r.cv2 ?? null,
      category: String(r.Category ?? r.category ?? ''),
      algorithm: String(r.Algorithm ?? r.algorithm ?? ''),
      isActive: r.IsActive ?? r.is_active ?? undefined,
      created_at:
        r.UpdatedAt ??
        r.CreatedAt ??
        r.created_at ??
        r.updated_at ??
        undefined,
    };
  }

  loadRows() {
    this.loading = true;
    this.errorMsg = null;
    this.rows = [];

    const params = new HttpParams()
      .set('period', this.slug())
      .set('include_inactive', String(this.includeInactive.value))
      .set('limit', '20000')
      .set('offset', '0');

    this.http.get<any[]>(`${this.API}/classify/saved`, { params }).subscribe({
      next: res => {
        this.rows = (res || []).map(x => this.toVM(x));

        // default sort by created_at desc if present, else ProductID asc
        if (this.rows.some(r => !!r.created_at)) {
          this.sortColumn = 'created_at';
          this.sortAsc = false;
          this.sortBy('created_at');
        } else {
          this.sortColumn = 'ProductID';
          this.sortAsc = true;
          this.sortBy('ProductID');
        }
      },
      error: e => {
        this.errorMsg = e?.error?.detail || e?.message || 'Failed to load saved classified results.';
      },
      complete: () => (this.loading = false)
    });
  }

  refresh() { this.loadRows(); }

  sortBy(col: keyof RowVM) {
    if (this.sortColumn === col) {
      this.sortAsc = !this.sortAsc;
    } else {
      this.sortColumn = col;
      this.sortAsc = true;
    }

    const dir = this.sortAsc ? 1 : -1;

    this.rows.sort((a, b) => {
      const av = (a as any)[col];
      const bv = (b as any)[col];

      if (col === 'adi' || col === 'cv2') {
        // Put nulls last (or first depending on direction)
        const na = av == null ? Number.POSITIVE_INFINITY : Number(av);
        const nb = bv == null ? Number.POSITIVE_INFINITY : Number(bv);
        return (na - nb) * dir;
      }

      if (col === 'created_at') {
        const da = av ? new Date(av).getTime() : 0;
        const db = bv ? new Date(bv).getTime() : 0;
        return (da - db) * dir;
      }

      return String(av ?? '').localeCompare(String(bv ?? '')) * dir;
    });
  }

  exportCSV() {
    if (!this.rows.length) return;

    const header = [
      'ProductID',
      'ChannelID',
      'LocationID',
      'Period',
      'ADI',
      'CV2',
      'Category',
      'Algorithm',
      'IsActive',
      'CreatedAt'
    ];

    const lines = this.rows.map(r =>
      [
        r.ProductID,
        r.ChannelID,
        r.LocationID,
        r.Period,
        r.adi ?? '',
        r.cv2 ?? '',
        r.category,
        r.algorithm,
        r.isActive ?? '',
        r.created_at ?? ''
      ]
        .map(v => `"${String(v).replace(/"/g, '""')}"`)
        .join(',')
    );

    const csv = [header.join(','), ...lines].join('\r\n');

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `Classified_Results_${this.period.value}${this.includeInactive.value ? '_with_inactive' : ''}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();

    URL.revokeObjectURL(url);
  }

  runInPlanning() {
    // Adjust route if your actual path differs
    this.router.navigate(['/planning-run/classify-forecast-elements'], {
      queryParams: { period: this.period.value }
    });
  }
}
