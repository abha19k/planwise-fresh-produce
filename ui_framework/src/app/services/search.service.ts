import { inject, Injectable, signal } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';


export interface KeyTriplet { ProductID: string; ChannelID: string; LocationID: string; }
export interface SearchResult { query: string; count: number; keys: KeyTriplet[]; }
export interface SavedSearch { id?: number; name: string; query: string; created_at?: string; }


@Injectable({ providedIn: 'root' })
export class SearchService {
private http = inject(HttpClient);
private base = 'http://127.0.0.1:8000/api'; // adjust if backend served elsewhere


result = signal<SearchResult | null>(null);
loading = signal(false);
error = signal<string | null>(null);


async run(q: string, limit = 5000, offset = 0) {
this.loading.set(true); this.error.set(null);
const params = new HttpParams().set('q', q).set('limit', limit).set('offset', offset);
try {
const res = await this.http.get<SearchResult>(`${this.base}/search`, { params }).toPromise();
this.result.set(res!);
} catch (e: any) {
this.error.set(e?.message || 'Search failed');
} finally {
this.loading.set(false);
}
}


listSaved() { return this.http.get<SavedSearch[]>(`${this.base}/saved-searches`); }
save(item: SavedSearch) { return this.http.post(`${this.base}/saved-searches`, item); }
}