const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4, 5], [5, 1]);
const ys = tf.tensor2d([2, 4, 6, 8, 10], [5, 1]);

// Train the model using the data.
model.fit(xs, ys, {epochs: 500}).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  var a = model.predict(tf.tensor2d([6], [1, 1]));
  document.write(a);
  // Open the browser devtools to see the output
});