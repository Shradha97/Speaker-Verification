$(document).ready(function(){
    console.log('document ready!')
    // connect to the socket server.
    var socket = io.connect('http://' + document.domain + ':' + location.port + '/test');
    var session = null, trainTab=false, train=null;

    //////////////////////////
    //    Layout setup
    //////////////////////////

	function populateMenu(tab){
		if (tab=='infer'){
			$('#trainTabButton')[0].innerHTML='Go to: Enrollment';
			$('#menuId')[0].innerHTML=' \
			    <button type="button" id="recordClip"> Record voice clip </button> \
                <button type="button" id="identifyPerson"> Identify Person </button>';
		}
		else if(tab=='enroll'){
			$('#trainTabButton')[0].innerHTML='Go to: Inference';
			$('#menuId')[0].innerHTML = ' \
				<button type="button" id="recordClip"> Record voice clip </button> \
				<input type="text" id="personName" placeholder="Name" /> \
				<button type="button" id="enrollPerson"> Enroll Person </button>';
            
            train = {'id':null,'labels':null,'images':null} // Saket Note: Correctly initialize, may not need all this
		}
	}
	////  Default run tab opened
	populateMenu('enroll');


	//////////////////////////
	//    Button listeners
	//////////////////////////

	// To toggle train/run tabs
	$(document).on('click','#trainTabButton',function(){
		console.log('[#trainTabButton] clicked');
		if (trainTab == false){
			trainTab = true;
			populateMenu('enroll');
			socket.emit('trainReq',{'type':'trainOn'});
		}
		else{
			trainTab = false;
			populateMenu('infer');
			socket.emit('trainReq',{'type':'trainOff'});
		}
	});

    $(document).on('click','#recordClip',function(){
        console.log('[#recordClip] clicked');
        data = {'type':'recordClip'};
        socket.emit('req',data);
    });

    $(document).on('click','#enrollPerson',function(){
        console.log('[#enrollPerson] clicked');
        data = {'type':'enrollPerson'};
        socket.emit('req',data);
    });

    $(document).on('click','#identifyPerson',function(){
        console.log('[#identifyPerson] clicked');
        data = {'type':'identifyPerson'};
        socket.emit('req',data);
    });

});