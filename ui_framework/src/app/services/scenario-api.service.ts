import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Scenario } from './scenario.service';

@Injectable({
  providedIn: 'root'
})
export class ScenarioApiService {

  private http = inject(HttpClient);

  private readonly API = 'http://127.0.0.1:8000/api';

  getScenarios(dbSchema: string): Observable<Scenario[]> {

    const params = new HttpParams()
      .set('db_schema', dbSchema);

    return this.http.get<Scenario[]>(`${this.API}/scenarios`, { params });
  }

  copyScenario(
    parentScenarioId: number,
    dbSchema: string,
    name: string,
    createdBy?: string | null
  ): Observable<Scenario> {

    const params = new HttpParams()
      .set('db_schema', dbSchema);

    return this.http.post<Scenario>(
      `${this.API}/scenarios/${parentScenarioId}/copy`,
      {
        name,
        created_by: createdBy ?? null
      },
      { params }
    );
  }
}
