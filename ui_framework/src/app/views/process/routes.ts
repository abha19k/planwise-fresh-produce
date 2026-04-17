import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    data: {
      title: 'Process'
    },
    children: [
      {
        path: '',
        redirectTo: 'scenario-manager',
        pathMatch: 'full'
      },
      {
        path: 'scenario-manager',
        loadComponent: () =>
          import('./scenario-manager/scenario-manager.component').then(m => m.ScenarioManagerComponent),
        data: {
          title: 'Scenario Manager'
        }
      },
      {
        path: 'forecast-tuning',
        loadComponent: () =>
          import('./forecast-tuning/forecast-tuning.component').then(m => m.ForecastTuningComponent),
        data: {
          title: 'Forecast Tuning'
        }
      }
      // {
      //   path: 'consensus-forecasting',
      //   loadComponent: () => import('./consensus-forecasting/consensus-forecasting.component').then(m => m.ConsensusForecastingComponent),
      //   data: {
      //     title: 'Consensus Forecasting'
      //   }
      // }
    ]
  }
];