let alphabet = []
var stop = false
function dummy(){}
function __init__(){
	var editor = ace.edit("code");
    editor.setTheme("ace/theme/tomorrow_night_eighties");
    editor.session.setMode("ace/mode/julia");

	alphabet = ['\n', '\t', '\n', ...(new Array(95)).fill(0).map((e, i) => String.fromCharCode(i + 32)), dummy];
	model.reset();
	var seed = randomLetter(alphabet);
	appendtext(model, alphabet, editor, 0, seed);

	document.querySelector("#editor .replay").addEventListener("click", function(event){
		console.log("clicked")
		stop = true;
		editor.setValue("", 1);
		setTimeout(()=>{
			stop = false;
			appendtext(model, alphabet, editor, 0, seed);
		}, 100)
	})

}

async function appendtext(m, alphabet, target, i, seed){
	if(stop || i > 100)return;
	var text = await sample(m, alphabet, 5, seed);
	seed = text.slice(-1)
	target.setValue(target.getValue() + text, 1);
	requestAnimationFrame(() => appendtext(m, alphabet, target, i + 1, seed))
}

async function sample(m, alphabet, len, c){
	var buf = ""
  	for(i = 1; i<=len; i++){
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
		if(arr[end] <= 0|0) return end;
		if(arr[start] > 0|0) return start;
		
		mid = Math.floor((start + end)/2);
		if (arr[mid] <= 0|0){
			start = mid + 1;
		}
		else{
			end = mid;
		}
	}
}

async function wsample(alphabet, dist){
	dist = await tf.sub(dist.cumsum(), tf.scalar(Math.random())).data();
	var i = findpos(dist);
	return alphabet[i]
}