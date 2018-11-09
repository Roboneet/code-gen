(function(obj){

	obj.__init__ = __init__;

	obj.key = 0;
	function dummy(){}
	const alphabet = ['\n', '\t', '\n', ...(new Array(95)).fill(0).map((e, i) => String.fromCharCode(i + 32)), dummy];

	function __init__(){
		var editor = ace.edit("code");
	    editor.setTheme("ace/theme/tomorrow_night_eighties");
	    editor.session.setMode("ace/mode/julia");
		
		var gen = new CodeGen(editor, model, alphabet, obj);
		document.querySelector("#editor .replay").addEventListener("click", function(event){
			obj.key++;
			gen.setup();
			setTimeout(gen.play.bind(gen), 100);
		})

		gen.setup();
		gen.play();
	}

	function CodeGen(editor, model, alphabet, refobj, batchlen=5, maxlen=100){
		this.target = editor;
		this.model = model;
		this.alphabet = alphabet;
		this.seed = "";
		this.batchlen = batchlen;
		this.maxlen = maxlen;
		this.refobj = refobj;
	}

	CodeGen.prototype.setup = function(){
		this.target.setValue("", 1);
		this.model.reset();
		this.seed = randomLetter(alphabet);
	}

	CodeGen.prototype.play = function() {
		console.log(this)
		return this.appendtext(0, this.refobj.key);
	};

	CodeGen.prototype.appendtext = async function(i, key){
		if(key != this.refobj.key || i > this.maxlen)return;
		var text = await sample(this.model, this.alphabet, this.batchlen, this.seed);
		this.target.setValue(this.target.getValue() + text, 1);
		this.seed = text.slice(-1);
		next = () => this.appendtext(i + 1, key);
		return requestAnimationFrame(next);
	}

	async function sample(m, alphabet, len, c){
		var buf = "";
		var inp__, inp;
	  	for(var i = 1; i<=len; i++){
		    inp__ = tf.tensor([alphabet.indexOf(c)], [1], 'int32');
		    inp = tf.oneHot(inp__, alphabet.length).unstack()[0].toFloat();
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