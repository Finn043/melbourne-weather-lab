import type { BrowserModel, WeatherDay, WeatherDataset } from './types';

const DAY_MS = 86_400_000;

function dayOfYear(date: string): number {
  const parsed = new Date(`${date}T00:00:00Z`);
  return Math.floor((parsed.getTime() - Date.UTC(parsed.getUTCFullYear(), 0, 1)) / DAY_MS) + 1;
}

function isLeapYear(year: number): boolean {
  return year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
}

export function featureVector(day: WeatherDay, startYear: number, endYear: number): number[] {
  const date = new Date(`${day.date}T00:00:00Z`);
  const year = date.getUTCFullYear();
  const yearLength = isLeapYear(year) ? 366 : 365;
  const yearPosition = (year - startYear) / Math.max(1, endYear - startYear);
  const dayPosition = (2 * Math.PI * dayOfYear(day.date)) / yearLength;
  const yearAngle = 2 * Math.PI * yearPosition;
  return [day.tmin, day.tmax, day.prcp, Math.sin(dayPosition), Math.cos(dayPosition), Math.sin(yearAngle), Math.cos(yearAngle)];
}

function runLstmLayer(sequence: number[][], layer: BrowserModel['lstm'][number]): number[][] {
  const hidden = Array<number>(layer.units).fill(0);
  let cell = Array<number>(layer.units).fill(0);
  const outputs: number[][] = [];
  for (const input of sequence) {
    const values = layer.bias.map((bias, out) => {
      let sum = bias;
      for (let i = 0; i < input.length; i += 1) sum += input[i] * layer.kernel[i][out];
      for (let i = 0; i < hidden.length; i += 1) sum += hidden[i] * layer.recurrent[i][out];
      return sum;
    });
    const nextCell = Array<number>(layer.units);
    const nextHidden = Array<number>(layer.units);
    for (let unit = 0; unit < layer.units; unit += 1) {
      const sigmoid = (value: number) => 1 / (1 + Math.exp(-value));
      const inputGate = sigmoid(values[unit]);
      const forgetGate = sigmoid(values[layer.units + unit]);
      const candidate = Math.tanh(values[layer.units * 2 + unit]);
      const outputGate = sigmoid(values[layer.units * 3 + unit]);
      nextCell[unit] = forgetGate * cell[unit] + inputGate * candidate;
      nextHidden[unit] = outputGate * Math.tanh(nextCell[unit]);
    }
    cell = nextCell;
    hidden.splice(0, hidden.length, ...nextHidden);
    outputs.push(nextHidden);
  }
  return outputs;
}

function dense(vector: number[], layer: BrowserModel['dense'][number]): number[] {
  return layer.bias.map((bias, out) => {
    let value = bias;
    for (let i = 0; i < vector.length; i += 1) value += vector[i] * layer.kernel[i][out];
    return layer.activation === 'relu' ? Math.max(0, value) : value;
  });
}

export function predictRainfall(history: WeatherDay[], dataset: Pick<WeatherDataset, 'startYear' | 'endYear'>, model: BrowserModel): number {
  if (history.length < model.lookback) throw new Error(`At least ${model.lookback} days of observations are needed.`);
  let sequence = history.slice(-model.lookback).map((day) =>
    featureVector(day, dataset.startYear, dataset.endYear).map((value, index) => (value - model.featureMean[index]) / model.featureScale[index]),
  );
  for (const layer of model.lstm) sequence = runLstmLayer(sequence, layer);
  let output = sequence[sequence.length - 1];
  for (const layer of model.dense) output = dense(output, layer);
  const logRainfall = (output[0] - model.targetMin[0]) / model.targetScale[0];
  return Math.max(0, Math.expm1(logRainfall));
}

export function predictMaximumTemperature(minimum: number, precipitation: number, model: BrowserModel): number {
  const [minCoefficient, rainCoefficient] = model.regression.coefficients;
  return minCoefficient * minimum + rainCoefficient * precipitation + model.regression.intercept;
}

export function classifyWeather(day: Pick<WeatherDay, 'tmin' | 'tmax' | 'prcp'>, model: BrowserModel): BrowserModel['clusters']['groups'][number] {
  const values = [day.tmin, day.tmax, day.prcp].map((value, index) => (value - model.clusters.mean[index]) / model.clusters.scale[index]);
  return model.clusters.groups.reduce((closest, group) => {
    const distance = group.center.reduce((sum, center, index) => sum + (values[index] - center) ** 2, 0);
    const closestDistance = closest.center.reduce((sum, center, index) => sum + (values[index] - center) ** 2, 0);
    return distance < closestDistance ? group : closest;
  });
}
