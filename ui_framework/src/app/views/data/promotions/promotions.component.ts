import { CommonModule } from '@angular/common';
import { Component, OnInit, inject } from '@angular/core';
import {
  FormBuilder,
  FormGroup,
  ReactiveFormsModule,
  Validators
} from '@angular/forms';
import { HttpClient, HttpParams } from '@angular/common/http';
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

import { ScenarioService } from '../../../services/scenario.service';

interface PromotionRow {
  PromoID: string;
  PromoName: string | null;
  StartDate: string | null;
  EndDate: string | null;
  ProductID: string | null;
  ChannelID: string | null;
  LocationID: string | null;
  PromoLevel: number | string | null;
  DiscountPct: number | string | null;
  UpliftPct: number | string | null;
  Notes: string | null;
}

interface PromotionsApiResponse {
  scenario_id: number;
  count: number;
  rows: PromotionRow[];
}

interface ProductRow { ProductID: string; }
interface ChannelRow { ChannelID: string; }
interface LocationRow { LocationID: string; }

@Component({
  standalone: true,
  selector: 'app-promotions',
  templateUrl: './promotions.component.html',
  styleUrls: ['./promotions.component.scss'],
  imports: [
    CommonModule,
    ReactiveFormsModule,
    
    TextColorDirective,
    CardComponent,
    CardBodyComponent,
    CardHeaderComponent,
    CardFooterComponent,
    RowComponent,
    ColComponent,
    ButtonDirective,
    TableDirective
  ]
})
export class PromotionsComponent implements OnInit {
  private http = inject(HttpClient);
  private fb = inject(FormBuilder);

  readonly scenarioService = inject(ScenarioService);

  private readonly API = 'http://127.0.0.1:8000/api';
  private readonly DB_SCHEMA = 'planwise_fresh_produce';
  private readonly UPDATED_BY = 'abha';

  productIds: string[] = [];
  channelIds: string[] = [];
  locationIds: string[] = [];

  filterForm: FormGroup = this.fb.group({
    productid: [''],
    channelid: [''],
    locationid: [''],
    date_from: ['', Validators.required],
    date_to: ['', Validators.required]
  });

  editForm: FormGroup = this.fb.group({
    PromoID: ['', Validators.required],
    PromoName: [''],
    StartDate: ['', Validators.required],
    EndDate: ['', Validators.required],
    ProductID: [''],
    ChannelID: [''],
    LocationID: [''],
    PromoLevel: [null],
    DiscountPct: [null],
    UpliftPct: [null],
    Notes: ['']
  });

  rows: PromotionRow[] = [];
  totalRows = 0;

  loading = false;
  saving = false;
  loadingDropdowns = false;
  deletingPromoId: string | null = null;
  errorMessage: string | null = null;
  successMessage: string | null = null;

  isEditing = false;
  isNewRow = false;
  originalPromoId: string | null = null;

  ngOnInit(): void {
    const today = new Date();
    const yyyy = today.getFullYear();
    const mm = String(today.getMonth() + 1).padStart(2, '0');

    this.filterForm.patchValue({
      date_from: `${yyyy}-${mm}-01`
    });

    const end = new Date(today.getFullYear(), today.getMonth() + 1, 0);
    const endY = end.getFullYear();
    const endM = String(end.getMonth() + 1).padStart(2, '0');
    const endD = String(end.getDate()).padStart(2, '0');

    this.filterForm.patchValue({
      date_to: `${endY}-${endM}-${endD}`
    });

    this.loadDropdowns();
  }

  get currentScenarioId(): number {
    return this.scenarioService.selectedScenarioId();
  }

  private schemaParams(): HttpParams {
    return new HttpParams().set('db_schema', this.DB_SCHEMA);
  }

  private baseParams(): HttpParams {
    return new HttpParams()
      .set('db_schema', this.DB_SCHEMA)
      .set('scenario_id', this.currentScenarioId);
  }

  private normalizeNullable(value: any): any {
    if (value === '' || value === undefined) return null;
    return value;
  }

  private buildPromotionPayloadFromForm(): PromotionRow {
    const raw = this.editForm.getRawValue();

    return {
      PromoID: String(raw.PromoID || '').trim(),
      PromoName: this.normalizeNullable(raw.PromoName),
      StartDate: this.normalizeNullable(raw.StartDate),
      EndDate: this.normalizeNullable(raw.EndDate),
      ProductID: this.normalizeNullable(raw.ProductID),
      ChannelID: this.normalizeNullable(raw.ChannelID),
      LocationID: this.normalizeNullable(raw.LocationID),
      PromoLevel: raw.PromoLevel === '' ? null : raw.PromoLevel,
      DiscountPct: raw.DiscountPct === '' ? null : raw.DiscountPct,
      UpliftPct: raw.UpliftPct === '' ? null : raw.UpliftPct,
      Notes: this.normalizeNullable(raw.Notes)
    };
  }

  loadDropdowns(): void {
    this.loadingDropdowns = true;

    Promise.all([
      this.http.get<ProductRow[]>(`${this.API}/products`, { params: this.schemaParams() }).toPromise(),
      this.http.get<ChannelRow[]>(`${this.API}/channels`, { params: this.schemaParams() }).toPromise(),
      this.http.get<LocationRow[]>(`${this.API}/locations`, { params: this.schemaParams() }).toPromise()
    ])
      .then(([products, channels, locations]) => {
        this.productIds = (products ?? []).map(r => String(r.ProductID)).filter(Boolean);
        this.channelIds = (channels ?? []).map(r => String(r.ChannelID)).filter(Boolean);
        this.locationIds = (locations ?? []).map(r => String(r.LocationID)).filter(Boolean);
        this.loadingDropdowns = false;
      })
      .catch((err) => {
        console.error('Dropdown load failed', err);
        this.errorMessage = err?.error?.detail || err?.message || 'Failed to load master data.';
        this.loadingDropdowns = false;
      });
  }

  loadPromotions(): void {
    this.errorMessage = null;
    this.successMessage = null;

    if (this.filterForm.invalid) {
      this.errorMessage = 'Please select both date_from and date_to.';
      this.rows = [];
      this.totalRows = 0;
      return;
    }

    this.loading = true;

    const raw = this.filterForm.getRawValue();

    let params = this.baseParams()
      .set('date_from', raw.date_from)
      .set('date_to', raw.date_to);

    const productid = String(raw.productid || '').trim();
    const channelid = String(raw.channelid || '').trim();
    const locationid = String(raw.locationid || '').trim();

    if (productid) params = params.set('productid', productid);
    if (channelid) params = params.set('channelid', channelid);
    if (locationid) params = params.set('locationid', locationid);

    this.http.get<PromotionsApiResponse>(`${this.API}/promotions`, { params }).subscribe({
      next: (res) => {
        this.rows = Array.isArray(res?.rows) ? res.rows : [];
        this.totalRows = Number(res?.count ?? this.rows.length);
        this.loading = false;
      },
      error: (err) => {
        console.error('Promotions load failed', err);
        this.errorMessage = err?.error?.detail || err?.message || 'Failed to load promotions.';
        this.rows = [];
        this.totalRows = 0;
        this.loading = false;
      }
    });
  }

  clearFilters(): void {
    const current = this.filterForm.getRawValue();

    this.filterForm.patchValue({
      productid: '',
      channelid: '',
      locationid: '',
      date_from: current.date_from,
      date_to: current.date_to
    });

    this.rows = [];
    this.totalRows = 0;
    this.errorMessage = null;
    this.successMessage = null;
  }

  startAdd(): void {
    this.errorMessage = null;
    this.successMessage = null;
    this.isEditing = true;
    this.isNewRow = true;
    this.originalPromoId = null;

    this.editForm.reset({
      PromoID: '',
      PromoName: '',
      StartDate: this.filterForm.get('date_from')?.value || '',
      EndDate: this.filterForm.get('date_to')?.value || '',
      ProductID: this.filterForm.get('productid')?.value || '',
      ChannelID: this.filterForm.get('channelid')?.value || '',
      LocationID: this.filterForm.get('locationid')?.value || '',
      PromoLevel: null,
      DiscountPct: null,
      UpliftPct: null,
      Notes: ''
    });
  }

  startEdit(row: PromotionRow): void {
    this.errorMessage = null;
    this.successMessage = null;
    this.isEditing = true;
    this.isNewRow = false;
    this.originalPromoId = row.PromoID;

    this.editForm.patchValue({
      PromoID: row.PromoID ?? '',
      PromoName: row.PromoName ?? '',
      StartDate: row.StartDate ?? '',
      EndDate: row.EndDate ?? '',
      ProductID: row.ProductID ?? '',
      ChannelID: row.ChannelID ?? '',
      LocationID: row.LocationID ?? '',
      PromoLevel: row.PromoLevel ?? null,
      DiscountPct: row.DiscountPct ?? null,
      UpliftPct: row.UpliftPct ?? null,
      Notes: row.Notes ?? ''
    });
  }

  duplicateToNew(row: PromotionRow): void {
    this.errorMessage = null;
    this.successMessage = null;
    this.isEditing = true;
    this.isNewRow = true;
    this.originalPromoId = null;

    this.editForm.patchValue({
      PromoID: `${row.PromoID}_COPY`,
      PromoName: row.PromoName ?? '',
      StartDate: row.StartDate ?? '',
      EndDate: row.EndDate ?? '',
      ProductID: row.ProductID ?? '',
      ChannelID: row.ChannelID ?? '',
      LocationID: row.LocationID ?? '',
      PromoLevel: row.PromoLevel ?? null,
      DiscountPct: row.DiscountPct ?? null,
      UpliftPct: row.UpliftPct ?? null,
      Notes: row.Notes ?? ''
    });
  }

  cancelEdit(): void {
    this.isEditing = false;
    this.isNewRow = false;
    this.originalPromoId = null;
    this.editForm.reset();
  }

  savePromotion(): void {
    this.errorMessage = null;
    this.successMessage = null;

    if (this.editForm.invalid) {
      this.errorMessage = 'Please fill the required fields.';
      this.editForm.markAllAsTouched();
      return;
    }

    const row = this.buildPromotionPayloadFromForm();
    if (!row.PromoID) {
      this.errorMessage = 'PromoID is required.';
      return;
    }

    this.saving = true;

    const pkPromoId = this.originalPromoId || row.PromoID;

    const body = {
      table_name: 'promotions',
      pk: {
        PromoID: pkPromoId
      },
      row,
      is_deleted: false,
      updated_by: this.UPDATED_BY
    };

    this.http.post<any>(
      `${this.API}/scenarios/${this.currentScenarioId}/override`,
      body,
      { params: this.schemaParams() }
    ).subscribe({
      next: () => {
        this.saving = false;
        this.successMessage = this.isNewRow
          ? 'Promotion added to selected scenario.'
          : 'Promotion updated for selected scenario.';
        this.cancelEdit();
        this.loadPromotions();
      },
      error: (err) => {
        console.error('Promotion save failed', err);
        this.errorMessage = err?.error?.detail || err?.message || 'Failed to save promotion.';
        this.saving = false;
      }
    });
  }

  deletePromotion(row: PromotionRow): void {
    this.errorMessage = null;
    this.successMessage = null;

    const promoId = String(row.PromoID || '').trim();
    if (!promoId) {
      this.errorMessage = 'PromoID is required to delete a promotion.';
      return;
    }

    this.deletingPromoId = promoId;

    const body = {
      table_name: 'promotions',
      pk: {
        PromoID: promoId
      },
      row: null,
      is_deleted: true,
      updated_by: this.UPDATED_BY
    };

    this.http.post<any>(
      `${this.API}/scenarios/${this.currentScenarioId}/override`,
      body,
      { params: this.schemaParams() }
    ).subscribe({
      next: () => {
        this.successMessage = 'Promotion deleted from selected scenario.';
        this.deletingPromoId = null;
        this.loadPromotions();
      },
      error: (err) => {
        console.error('Promotion delete failed', err);
        this.errorMessage = err?.error?.detail || err?.message || 'Failed to delete promotion.';
        this.deletingPromoId = null;
      }
    });
  }

  exportToCSV(): void {
    if (!this.rows.length) return;

    const header = [
      'PromoID',
      'PromoName',
      'StartDate',
      'EndDate',
      'ProductID',
      'ChannelID',
      'LocationID',
      'PromoLevel',
      'DiscountPct',
      'UpliftPct',
      'Notes'
    ];

    const body = this.rows.map(r =>
      header.map(h => `"${String((r as any)[h] ?? '').replace(/"/g, '""')}"`).join(',')
    );

    const csv = [header.join(','), ...body].join('\r\n');

    const raw = this.filterForm.getRawValue();
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `Promotions_s${this.currentScenarioId}_${raw.date_from}_${raw.date_to}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
}