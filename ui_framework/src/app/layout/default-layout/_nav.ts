import { INavData } from '@coreui/angular';

export const navItems: INavData[] = [
  {
    name: 'Dashboard',
    url: '/dashboard',
    iconComponent: { name: 'cil-speedometer' },
    // badge: {
    //   color: 'info',
    //   text: 'NEW'
    // }
  }, 
  {
    name: 'Data',
    url: '/data',
    iconComponent: { name: 'cil-speedometer' },
    children: [
      {
        name: 'Product',
        url: '/data/product',
        icon: 'nav-icon-bullet'
      },
      {
        name: 'Channel',
        url: '/data/channel',
        icon: 'nav-icon-bullet'
      },
      {
        name: 'Location',
        url: '/data/location',
        icon: 'nav-icon-bullet'
      },
      {
        name: 'History',
        url: '/data/history',
        icon: 'nav-icon-bullet'
      },
      {
        name: 'Forecast Element',
        url: '/data/forecast-element',
        icon: 'nav-icon-bullet'
      },
      {
        name: 'Classified Forecast Elements',
        url: '/data/classified-forecast-elements',
        icon: 'nav-icon-bullet'
      },
      {
        name: 'Forecast',
        url: '/data/forecast',
        icon: 'nav-icon-bullet'
      },
      {
        name: 'KPI',
        url: '/data/kpi',
        icon: 'nav-icon-bullet'
      }
      // {
      //   name: 'Weather Correlation',
      //   url: '/data/weather-correlation',
      //   icon: 'nav-icon-bullet'
      // },
      // {
      //   name: 'Supply Data',
      //   url: '/data/supply-data',
      //   icon: 'nav-icon-bullet' 
      // },      
      // {
      //   name: 'Global Parameters',
      //   url: '/data/global-parameters',
      //   icon: 'nav-icon-bullet'
      // }
    ]
  },
  {
    name: 'Process',
    url: '/process',
    iconComponent: { name: 'cil-speedometer' },
    children: [
      {
        name: 'Consensus Forecasting',
        url: '/process/consensus-forecasting',
        icon: 'nav-icon-bullet'
      },
      {
        name: 'Forecast Tuning',
        url: '/process/forecast-tuning',
        icon: 'nav-icon-bullet'
      }
    ]
  },
  {
    name: 'Planning Run',
    url: '/planning-run',
    iconComponent: { name: 'cil-speedometer' },
    children: [
      // {
      //   name: 'Load History',
      //   url: '/planning-run/load-history',
      //   icon: 'nav-icon-bullet'
      // },
      {
        name: 'Cleanse History',
        url: '/planning-run/cleanse-history',
        icon: 'nav-icon-bullet'
      },
      {
        name: 'Classify Forecast Elements',
        url: '/planning-run/classify-forecast-elements',
        icon: 'nav-icon-bullet'
      }
    ]
   
  },
  {
    name: 'Search',
    url: '/search',
    iconComponent: { name: 'cil-speedometer' },
  }      
];
