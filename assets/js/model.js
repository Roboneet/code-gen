let model = (function () {
  let math = tf;
  model.weights = [];
  let init = [];
  let states = [];
  function spider(cat) {
    return math.add(math.vectorTimesMatrix(cat, model.weights[0]), model.weights[1]);
  };
  function kangaroo(albatross) {
    return albatross;
  };
  function kudu(falcon, hornet) {
    return flux.iterate(falcon);
  };
  function owl(porpoise, hamster, cobra) {
    return flux.iterate(porpoise, cobra);
  };
  function mandrill(herring) {
    let bee = states[1];
    let cassowary = kudu(bee, 1);
    let chicken = cassowary[0];
    let ant = math.add(math.add(math.vectorTimesMatrix(herring, model.weights[2]), math.vectorTimesMatrix(chicken, model.weights[3])), model.weights[4]);
    let pheasant = chicken[String("shape")];
    let donkey = pheasant[(pheasant[String("length")]-1)];
    let chimpanzee = tf.keep(math.add(math.mul(math.sigmoid(math.slice(ant, (donkey*1), donkey)), owl(bee, 2, cassowary[1])[0]), math.mul(math.sigmoid(math.slice(ant, (donkey*0), donkey)), math.tanh(math.slice(ant, (donkey*2), donkey)))));
    let goose = tf.keep(math.mul(math.sigmoid(math.slice(ant, (donkey*3), donkey)), math.tanh(chimpanzee)));
    tf.dispose(bee[0]);
    tf.dispose(bee[1]);  
    states[1] = [goose, chimpanzee];
    return goose;
  };
  function pigeon(dunlin) {
    return dunlin;
  };
  function raccoon(eland, gazelle) {
    return flux.iterate(eland);
  };
  function narwhal(parrot, eel, dinosaur) {
    return flux.iterate(parrot, dinosaur);
  };
  function pig(oyster) {
    let newt = states[0];
    let ape = raccoon(newt, 1);
    let magpie = ape[0];
    let turtle = math.add(math.add(math.vectorTimesMatrix(oyster, model.weights[5]), math.vectorTimesMatrix(magpie, model.weights[6])), model.weights[7]);
    let lyrebird = magpie[String("shape")];
    let tiger = lyrebird[(lyrebird[String("length")]-1)];
    let waterbuffalo = tf.keep(math.add(math.mul(math.sigmoid(math.slice(turtle, (tiger*1), tiger)), narwhal(newt, 2, ape[1])[0]), math.mul(math.sigmoid(math.slice(turtle, (tiger*0), tiger)), math.tanh(math.slice(turtle, (tiger*2), tiger)))));
    let dolphin = tf.keep(math.mul(math.sigmoid(math.slice(turtle, (tiger*3), tiger)), math.tanh(waterbuffalo)));
    tf.dispose(newt[0]);
    tf.dispose(newt[1]);
    states[0] = [dolphin, waterbuffalo];
    return dolphin;
  };
  function model(caribou) {
    return math.softmax(spider(kangaroo(mandrill(pigeon(pig(caribou)))))); 
  };
  model.reset = (function () {
    states = flux.deepcopy(init);
    return;
  });
  model.getStates = (function () {
    return states;
  });
  model.setWeights = (function (ws) {
    tf.dispose(init);
    tf.dispose(states);
    model.weights = ws;
    init = tf.keep([[model.weights[8], model.weights[9]], [model.weights[10], model.weights[11]]]);
    states = flux.deepcopy(init);
    __init__();
    return;
  });
  return model;
})();
flux.fetchWeights("./assets/bson/model.bson").then(model.setWeights);
