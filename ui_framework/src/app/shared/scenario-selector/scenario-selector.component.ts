import { CommonModule } from '@angular/common';
import { Component, OnInit, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import {
  CardBodyComponent,
  CardComponent,
  CardHeaderComponent,
  ColComponent,
  RowComponent
} from '@coreui/angular';

import { ScenarioApiService } from '../../services/scenario-api.service';
import { ScenarioService } from '../../services/scenario.service';

@Component({
  selector: 'app-scenario-selector',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    CardComponent,
    CardHeaderComponent,
    CardBodyComponent,
    RowComponent,
    ColComponent
  ],
  templateUrl: './scenario-selector.component.html'
})
export class ScenarioSelectorComponent implements OnInit {
  private scenarioApi = inject(ScenarioApiService);
  readonly scenarioService = inject(ScenarioService);

  dbSchema = 'planwise_fresh_produce';
  loading = false;
  error = '';

  ngOnInit(): void {
    this.loadScenarios();
  }

  loadScenarios(): void {
    this.loading = true;
    this.error = '';

    this.scenarioApi.getScenarios(this.dbSchema).subscribe({
      next: (items) => {
        this.scenarioService.setScenarios(items ?? []);
        this.loading = false;
      },
      error: (err) => {
        this.error = err?.error?.detail || 'Failed to load scenarios';
        this.loading = false;
      }
    });
  }

  onScenarioChange(value: string | number): void {
    const id = Number(value);
    if (!Number.isNaN(id) && id > 0) {
      this.scenarioService.setSelectedScenarioId(id);
    }
  }
}