(function(obj){

obj.__init__ = __init__;

let alphabet = []
var exec_key = 0;
function dummy(){}
function __init__(){
	var editor = ace.edit("code");
    editor.setTheme("ace/theme/tomorrow_night_eighties");
    editor.session.setMode("ace/mode/julia");
	alphabet = ['\n', '\t', '\n', ...(new Array(95)).fill(0).map((e, i) => String.fromCharCode(i + 32)), dummy];
	var play = (key) => appendtext(model, alphabet, editor, 0, setup(editor, model, alphabet), key)
	document.querySelector("#editor .replay").addEventListener("click", function(event){
		exec_key++;
		setTimeout(() => play(exec_key), 100);
	})
	return play(exec_key);
}

function setup(editor, model, alphabet){
	editor.setValue("", 1);
	model.reset();
	return randomLetter(alphabet);
}

async function appendtext(m, alphabet, target, i, seed, key){
	if(key != exec_key || i > 100)return;
	var text = await sample(m, alphabet, 5, seed);
	seed = text.slice(-1);
	target.setValue(target.getValue() + text, 1);
	return requestAnimationFrame(() => appendtext(m, alphabet, target, i + 1, seed, key))
}

async function sample(m, alphabet, len, c){
	var buf = ""
  	for(var i = 1; i<=len; i++){
	    var inp__ = tf.tensor([alphabet.indexOf(c)], [1], 'int32');
	    var inp = tf.oneHot(inp__, alphabet.length).unstack()[0].toFloat();
	    c = await wsample(alphabet, m(inp));
	    if(c == dummy)return buf + "\n"
	    buf += c;
  	}
  	return buf
}

function randomLetter(alphabet){
	return alphabet[Math.floor((alphabet.length)*Math.random())]
}

function findpos(arr, start, end){
	var l = arr.length;
	start = start || 0;
	end = end || l - 1;
	var mid;
	while(end >= start){
		if(end == start)return end;
		if(arr[end] <= 0)return end;
		if(arr[start] > 0)return start;
		
		mid = Math.floor((start + end)/2);
		if (arr[mid] <= 0)
			start = mid + 1;
		else
			end = mid;
	}
	return end;
}

async function wsample(alphabet, dist){
	dist = await tf.sub(dist.cumsum(), tf.scalar(Math.random())).data();
	var i = findpos(dist);
	return alphabet[i];
}

}(window))