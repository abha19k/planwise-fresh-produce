import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    data: {
      title: 'Data'
    },
    children: [
      {
        path: '',
        redirectTo: 'history',
        pathMatch: 'full'
      },
      {
        path: 'product',
        loadComponent: () => import('../data/product/product.component').then(m => m.ProductComponent),
        data: {
          title: 'Product'
        }
      },      
      {
        path: 'channel',
        loadComponent: () => import('./channel/channel.component').then(m => m.ChannelComponent),
        data: {
          title: 'Channel'
        }
      },
      {
        path: 'location',
        loadComponent: () => import('../data/location/location.component').then(m => m.LocationComponent),
        data: {
          title: 'Location'
        }
      },
      {
        path: 'history',
        loadComponent: () => import('./history/history.component').then(m => m.HistoryComponent),
        data: {
          title: 'History'
        }
      },      
      {
        path: 'forecast-element',
        loadComponent: () => import('../data/forecast-element/forecast-element.component').then(m => m.ForecastElementComponent),
        data: {
          title: 'Forecast Element'
        }
      },
      {
        path: 'classified-forecast-elements',
        loadComponent: () => import('./classified-forecast-elements/classified-forecast-elements.component').then(m => m.ClassifiedForecastElementsComponent),
        data: {
          title: 'Classified Forecast Elements'
        }
      },      
      {
        path: 'forecast',
        loadComponent: () => import('../data/forecast/forecast.component').then(m => m.ForecastComponent),
        data: {
          title: 'Forecast'
        }
      },
      {
        path: 'kpi',
        loadComponent: () => import('../data/kpi/kpi.component').then(m => m.KpiComponent),
        data: {
          title: 'KPI'
        }
      },
      // {
      //   path: 'kpi',
      //   loadComponent: () => import('../data/kpi/kpi-dashboard.component').then(m => m.KpiDashboardComponent),
      //   data: {
      //     title: 'KPI'
      //   }
      // },
      {
        path: 'weather-correlation',
        loadComponent: () => import('../data/weather-correlation/weather-correlation.component').then(m => m.WeatherCorrelationComponent),
        data: {
          title: 'Weather Correlation'
        }
      },
      {
        path: 'supply-data',
        loadComponent: () => import('../data/supply-data/supply-data.component').then(m => m.SupplyDataComponent),
        data: {
          title: 'Supply Data'
        } 
      }                       
    ]
  }
];
