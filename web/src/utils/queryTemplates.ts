// Query Templates for TwinQL Console
export interface QueryTemplate {
    id: string;
    name: string;
    description: string;
    query: string;
}

export const queryTemplates: QueryTemplate[] = [
    {
        id: 'single-comparison',
        name: 'Single Building Comparison',
        description: 'Compare baseline vs retrofit for one building',
        query: `-- Single Building Scenario Comparison
TWIN SELECT 
  h.value AS baseline,
  s.value AS retrofit,
  (h.value - s.value) AS saving
FROM 
  HISTORICAL twin_id='Twin_B123' 
    WINDOW '2018-01-01' TO '2019-01-01' 
    METRIC 'heat_load' 
    AGGREGATE BY month AS h,
  SIMULATE twin_id='Twin_B123' 
    SCENARIO deep_retrofit 
    MODEL 'BuildingHeat_v3'
    WINDOW '2018-01-01' TO '2019-01-01' 
    METRIC 'heat_load' 
    AGGREGATE BY month AS s`,
    },
    {
        id: 'historical-only',
        name: 'Historical Data Query',
        description: 'Fetch historical data for a building',
        query: `-- Historical Data Query
TWIN SELECT 
  h.value AS heat_load
FROM 
  HISTORICAL twin_id='Twin_B123' 
    WINDOW '2018-01-01' TO '2019-01-01' 
    METRIC 'heat_load' 
    AGGREGATE BY month AS h`,
    },
    {
        id: 'multi-scenario',
        name: 'Multi-Scenario Comparison',
        description: 'Compare multiple retrofit scenarios',
        query: `-- Multi-Scenario Comparison
TWIN SELECT 
  h.value AS baseline,
  s1.value AS medium_retrofit,
  s2.value AS deep_retrofit
FROM 
  HISTORICAL twin_id='Twin_B123' 
    WINDOW '2018-01-01' TO '2019-01-01' 
    METRIC 'heat_load' 
    AGGREGATE BY month AS h,
  SIMULATE twin_id='Twin_B123' 
    SCENARIO s_retrofit 
    MODEL 'BuildingHeat_v3'
    WINDOW '2018-01-01' TO '2019-01-01' 
    METRIC 'heat_load' 
    AGGREGATE BY month AS s1,
  SIMULATE twin_id='Twin_B123' 
    SCENARIO deep_retrofit 
    MODEL 'BuildingHeat_v3'
    WINDOW '2018-01-01' TO '2019-01-01' 
    METRIC 'heat_load' 
    AGGREGATE BY month AS s2`,
    },
    {
        id: 'define-scenario',
        name: 'Define New Scenario',
        description: 'Create a custom retrofit scenario',
        query: `-- Define a New Scenario
DEFINE SCENARIO custom_retrofit AS
  retrofit_package = {
    "wall_u_after": 0.20,
    "window_u_after": 1.0,
    "infiltration_after": 0.5
  }`,
    },
    {
        id: 'worst-performers',
        name: 'Find Worst Performers',
        description: 'Identify buildings with highest consumption',
        query: `-- Find Worst Performing Buildings
-- Note: This uses the Python API pattern
-- TwinQL for single building:
TWIN SELECT 
  h.value AS annual_consumption
FROM 
  HISTORICAL twin_id='Twin_B123' 
    WINDOW '2018-01-01' TO '2019-01-01' 
    METRIC 'heat_load' 
    AGGREGATE BY year AS h`,
    },
];
