import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    data: {
      title: 'Planning Run'
    },
    children: [
      {
        path: '',
        redirectTo: 'store-forecast',
        pathMatch: 'full'
      },
      {
        path: 'store-forecast',
        loadComponent: () => import('./store-forecast/store-forecast.component').then(m => m.StoreForecastComponent),
        data: {
          title: 'Store Forecast'
        }
      },
      // {
      //   path: 'load-history',
      //   loadComponent: () => import('./load-history/load-history.component').then(m => m.LoadHistoryComponent),
      //   data: {
      //     title: 'Load History'
      //   }
      // },
      {
      path: 'cleanse-history',
      loadComponent: () => import('./cleanse-history/cleanse-history.component').then(m => m.CleanseHistoryComponent),
      data: {
        title: 'Cleanse History'
      }
    },
    {
      path: 'classify-forecast-elements',
      loadComponent: () => import('./classify-forecast-elements/classify-forecast-elements.component').then(m => m.ClassifyForecastElementsComponent),
      data: {
        title: 'Classify Forecast Elements'
      }
    },
    // {
    //   path: 'forecast-tuning',
    //   loadComponent: () => import('./forecast-tuning/forecast-tuning.component').then(m => m.ForecastTuningComponent),
    //   data: {
    //     title: 'Forecast Tuning'
    //   }
    // }
    ]
  }
];
