(function(obj){

	obj.__init__ = __init__;

	obj.key = 0;
	function dummy(){}
	const alphabet = ['\n', '\t', '\n', ...(new Array(95)).fill(0).map((e, i) => String.fromCharCode(i + 32)), dummy];

	function __init__(){
		var editor = ace.edit("code");
	    editor.setTheme("ace/theme/tomorrow_night_eighties");
	    editor.session.setMode("ace/mode/julia");
		
		var gen = new CodeGen(model, alphabet, obj);
		var scr = new Screen(gen, editor);
		document.querySelector("#editor .replay").addEventListener("click", function(event){
			obj.key++;
			gen.play();
			setTimeout(scr.play.bind(scr), 100);
		})

		gen.play();
		scr.play();
	}

	function Screen(gen, editor){
		this.gen = gen;
		this.editor = editor;
	}

	Screen.prototype.play = function(){
		this.editor.setValue(this.gen.getVal(), 1);
		requestAnimationFrame(this.play.bind(this));
	}

	function CodeGen(model, alphabet, refobj, batchlen=5, maxlen=100){
		this.model = model;
		this.alphabet = alphabet;
		this.seed = "";
		this.batchlen = batchlen;
		this.maxlen = maxlen;
		this.refobj = refobj;
		this.buf = "";
	}

	CodeGen.prototype.getVal = function(){ return this.buf; }

	CodeGen.prototype.setup = function(){
		this.buf = "";
		this.model.reset();
		this.seed = randomLetter(alphabet);
	}

	CodeGen.prototype.play = function() {
		this.setup();
		return this.fillbuf(0, this.refobj.key);
	};

	CodeGen.prototype.fillbuf = async function(i, key){
		while(key == this.refobj.key && i < this.maxlen){
			var text = await sample(this.model, this.alphabet, this.batchlen, this.seed);
			this.buf += text;
			this.seed = text.slice(-1);
			i += 1;
		}
	}

	async function sample(m, alphabet, len, c){
		var buf = "";
		var out;
	  	for(var i = 1; i<=len; i++){
	  		out = tf.tidy(() =>{
			    var inp__ = tf.tensor([alphabet.indexOf(c)], [1], 'int32');
			    var inp = tf.oneHot(inp__, alphabet.length).unstack()[0].toFloat();
			    return m(inp)
			});
		    c = await wsample(alphabet, out);
		    tf.dispose(out);
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

	async function wsample(alphabet, dist){ // safe
		var rr = tf.tidy(() =>tf.sub(dist.cumsum(), tf.scalar(Math.random())));
		var d = await rr.data();
		tf.dispose(rr);
		var i = findpos(d);
		return alphabet[i];
	}

}(window))