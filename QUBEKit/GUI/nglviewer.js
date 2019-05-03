// Create NGL Stage object
var stage = new NGL.Stage( "viewport" );

// Handle window resizing
window.addEventListener( "resize", function( event ){
    stage.handleResize();
}, false );


function moleculeLoader(fileName){
    // If we have a state we need to get rid of it first when loading a new molecule
    console.log(stage.compList)
    if (stage.compList[0]){
        stage.removeComponent(stage.compList[0])
    }
    // check the file ext as mol is the same as sdf
    if (fileName.includes('.mol') && !fileName.includes('.mol2') ){
        stage.loadFile(fileName, {ext: "sdf"}).then(function (molecule) {
            molecule.addRepresentation("licorice", {multipleBond: "symmetric"});
            molecule.autoView();
        })
        return "Loaded";
    }else {
        stage.loadFile(fileName).then(function (molecule) {
            molecule.addRepresentation("licorice", {multipleBond: "symmetric"});
            molecule.autoView(); 
        })
        return "Loaded";
    }
}

function ChangeView(representation){
    stage.compList[0].removeRepresentation();
    stage.compList[0].addRepresentation(representation);
    stage.compList[0].autoView();
    return  'view changed to ' + representation;
}

function LoadCube(fileName, isolevel, color1=null, color2=null, opacity=0.7, opaqueBack=false){
    // first we should remove a componet by name
    stage.loadFile(fileName).then(function  (molecule){
        if (color1 && color2){
            // load two representations with the different colors
            molecule.addRepresentation("surface", {
                visible: true,
                isolevelType: "value",
                isolevel: isolevel,
                color: color1,
                opacity: opacity,
                opaqueBack: opaqueBack
              });
            molecule.addRepresentation("surface", {
                visible: true,
                isolevelType: "value",
                isolevel: isolevel,
                color: color2,
                opacity: opacity,
                opaqueBack: opaqueBack
              
        });
        molecule.autoView();
    }else {
        molecule.addRepresentation("surface", {
            visible: true,
            isolevel: 0.1, 
            opacity: 0.6
        });
        molecule.autoView();
    }
    return "cube loaded";
})
}

