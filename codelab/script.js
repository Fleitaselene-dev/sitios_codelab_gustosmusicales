console.log('Hello TensorFlow');


 // Cargar y limpiar los datos
 //Cargamos un conjunto de datos de automoviles
  const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
  const carsData = await carsDataResponse.json();
//Quitaremos las entradas que no tengan definidas las millas por galón ni la potencia.
  const cleaned = carsData.map(car => ({
    mpg: car.Miles_per_Gallon,
    horsepower: car.Horsepower,
  })).filter(car => (car.mpg != null && car.horsepower != null));
  
  return cleaned;


//Creacion del modelo secuencial
function createModel() {
  const model = tf.sequential();
  
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
  model.add(tf.layers.dense({ units: 1, useBias: true }));

  return model;
}

//Convertir los datos a tensores y normalizarlos

function convertToTensor(data) {
  return tf.tidy(() => {
    //Distribuye los datos de forma aleatoria
    tf.util.shuffle(data);
     //Convierte los datos a tensor
    const inputs = data.map(d => d.horsepower);
    const labels = data.map(d => d.mpg);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
    // Normaliza los datos usando rangos de 0-1 
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
}

//Entrenamiento al modelo
async function trainModel(model, inputs, labels) {
  model.compile({
    //Manejara las actualizaciones del modelo a medida que vea ejemplos
    optimizer: tf.train.adam(),
    //Indica al modelo los errores y que tan bien aprende
    //'meanSquaredError' comparar las predicciones que hizo el modelo con los valores verdaderos.
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });
  //tamaño datos que verá el modelo en cada iteración de entrenamiento
  const batchSize = 32;
  //ciclos de entrenamiento
  const epochs = 50;
 //inicio de bucle de entrenamiento
  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    ),
  });
}

//Prueba del modelo 
function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  const [xs, preds] = tf.tidy(() => {
    //Generamos 100 “ejemplos” nuevos para ingresarlos al modelo.
    const xs = tf.linspace(0, 1, 100);
    const preds = model.predict(xs.reshape([100, 1]));

    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => ({
    x: val,
    y: preds[i],
  }));

  const originalPoints = inputData.map(d => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    { name: 'Model Predictions vs Original Data' },
    { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300,
    }
  );
}

// Ejecutar el flujo completo al cargar la página
async function run() {
    //Obtener los datos
  const data = await getData();

  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg,
  }));
 // diagrama de dispersión de los datos
  tfvis.render.scatterplot(
    { name: 'Horsepower v MPG' },
    { values },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300,
    }
  );

    //Se creará una instancia del modelo y se mostrará un resumen de las capas en la página web.
  const model = createModel();
  tfvis.show.modelSummary({ name: 'Model Summary' }, model);
const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;

  await trainModel(model, inputs, labels);
  testModel(model, data, tensorData);
}

document.addEventListener('DOMContentLoaded', run);
