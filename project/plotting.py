import definitions as df

def plot():
    data =  [1.3688977906028819, 1.3380134311084948, 0, 0, 1.1953522845977582, 1.156570454502348, 1.1332699440588698, 1.1141540814294153, 1.0920082546445649, 1.0554679868531158, 1.006277318087545, 0, 0, 0.9678576503824888, 0, 
              0.9645279220792305, 0, 0, 0.9501889034984992, 0.8792428573576627, 0, 0.7996264651054213, 0.8102596588586504, 0.7889244295207912, 0.7145723211289835, 0.6788238473114798, 0.6888231923227494, 0, 0.6927012147690693, 
              0.6785095825654, 0, 0.7511628430675489, 0.7054426146500338, 0.7120614856682875, 0.7384177594536073, 0.6986068551281412, 0.6723041292318762, 0.6829307931339915, 0, 0, 0.7370977893434307, 0.7074499646589489, 0.6738984423571763, 
              0, 0, 0, 0, 0, 0, 0, 0, 0.6459594900692522, 0, 0, 0, 0, 0.4421656475543407, 0.43352752320167703, 0.4005235606324424, 0.41912384386289925, 0.3710967181143448, 0.3502300035734849, 0.33020855458961224, 0.3176333951239502, 0.35711379784405617, 
              0.37975902539736156, 0.38726954314022777, 0.3900048655044701, 0.37317037708094836, 0, 0, 0.58258614231499, 0.3505538761303315, 0.35954417849308623, 0.3782766819023881, 0.37886899592037676, 0.352038476364812, 0.3509293528363243, 0.3263414128490292, 
              0.3205793599183892, 0.30003345473272053, 0.28661179977924217, 0.28282690140232997, 0.2612362254727343, 0.2717782430829249, 0, 0, 0, 0, 0, 0.9894634708864838, 0.8776028537448155, 0.8603973606946757, 0.8485553870744083, 0.730198385313377, 
              0.6179364629742863, 0.528122064373126, 0.4648285367124619, 0.42706115394135136, 0.39035846684350645, 0.3657840229962771, 0.34522247263233063, 0.3613441428045409, 0.36081132409261707, 0.36959746287211814, 0.35138907258851404, 0.33044586443979934, 
              0.3448232239086592, 0.35505654488593447, 0.35397593410843015, 0.3499819422903898, 0.4069166486387435, 0.3619156285833279, 0.3516038281846982, 0.35334909434160944, 0.34896476362416473, 0.35667926599367206, 0.36121974558205244, 0.35651360443845487, 
              0.36354568081915684, 0.3693127726066706, 0.3939907636529824, 0.3908620017280571, 0.3829668754380334, 0.3843712635958539, 0.3212191065305664, 0.2690324154001192, 0.2797839239795288, 0.27636876213038614, 0.27249957060229907, 0.2888904976808641, 
              0.2929023933025156, 0.27857417810060686, 0.29325719668691386, 0.29217527147487865, 0.28991680505041995, 0.2987693504896389, 0.31152078122489796, 0.3499249449846461, 0, 0, 0.5579864971326408, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
              0.8095188811076501, 0.7752536109741694, 0.7479356900704327, 0.7314979366478173, 0.7524258889340596, 0.7586257849810466, 0.7231326795653084, 0.7226534583062827, 0.7222692421001917, 0.6505421605313827, 0.5758266179366289, 0.5383787296638509, 0.49409461739035443, 0.40467208818941924, 0.39088467042207076, 0.3789556590777685, 0.379017305263724, 0.3260888168832074, 0.29866379483902633, 0.30094282437427655, 0.2890935882226815, 0.2729127819387335, 0.25944267398378235, 0.2624915653567537, 0.270437109426673, 0.27653783889765965, 0.28425120500422857, 0.26419951647700557, 0.2727332600119154, 0.2608823818369999, 0.28585502667215273, 0.28773980439244157, 0.27395573507163473, 0.2704467010224064, 0.26149314780036087, 0.26051608456065367, 0.25954966903285565, 0.26703652008887846, 0.2643978963097515, 0.2408633331692987, 0.24558148800876733, 0.22998633035335186, 0.23217544841252896, 0.22592777609345513, 0.2521950391139309, 0.30145629897511883, 0.3900814850789395, 0.38200770215965196, 0.41849136625776, 0.5731214802774743, 0.6508686658788669, 0.6778326865213097, 0.7396551683666558, 0, 0, 0, 0, 1.0375166622310872, 0.5995077193381535, 0.5922494356807548, 0.602177557781013, 0.5984576359789718, 0.6781080690480812, 0.6846537074752856, 0.7049930432918632, 0.6642983199313524, 0.6607883867588124, 0.6469261501525317, 0.6766294676806732, 0.6796969403501119, 0.6852415078536451, 0.6911361757116957, 0.7025320852900857, 0.6974185270614848, 0.7354375372948345, 0.7159395118550901, 0.7257265769206727, 0.730650635830134, 0.7519013150894007, 0.7735650814551239, 0.7895890159796054, 0.824083967438486, 0.8667674442121831, 0.860933331647451, 0.9014166708639522, 0.9164027301589674, 0.8818108284394862, 0.8607392248401826, 0.8617970242327357, 0.8669468637504739, 0.8270339540691369, 0.8178805160605783, 0.8179317874482065, 0.8198022334264813, 0.8137172681215858, 0.7701808751659976, 0.8105004545450379, 0.787593733601769, 0.7651917628690144, 0.7501049398317365, 0.7296274690745909, 0.7107482580981787, 0.6991591008427169, 0.7005246376958193, 0.7089259024075556, 0.729573633355763, 0.7284486524658206, 0.7109442994611552, 0.695840671723319, 0.5526603979729622, 0.6098923341144037, 0.588611158766678, 0.6164710518038148, 0.594014491074166, 0.6233862549968517, 0.6204299735738951, 0.6022026638783361, 0.6047135524686735, 0.5998683544514376, 0.5826027371145144, 0.5988589509165589, 0.5987169666198374, 0.5994392632168272, 0.5828710504797268, 0.5946255770714783, 0.5992714351886497, 0.5886502024809267, 0.6005650342746567, 0.5933915559046701, 0.596882001626566, 0.5898007785042454, 0.6080149894480955, 0.6046080491866255, 0.5959390907893527, 0.6042977423098932, 0.6013132129168004, 0.625111967575806, 0.6386467669672078, 0.64080479296636, 0.6339945757049247, 0.6155062554189148, 0.6048017908258063, 0.6198605145857611, 0.6209099398692933, 0.6278615196040802, 0.6196757492020677, 0.6211948510771877, 0.6235855752360007, 0.6231371926182577, 0.6650099448117377, 0.6657471058349447, 0.7069643044259252, 0.6752409845150005, 0.6543904819060218, 0.6370183796317612, 0.6667474156443023, 0.6751967903411709, 0.6529105098897959, 0.658885378873986, 0.6662981409552526, 0.7539008736312041, 0.7972010289503357, 0.8287689215221923, 0.896555233080609, 0.9411800823568826, 1.004718652093769, 1.2114902615368137, 1.1343194123345823, 1.1081919324406384, 1.0385117564889315, 0.9077218508145471, 0.8451988131145912, 0.8664447341030472, 0.8346058365620652, 0.8290225708342951, 0.8603466258881106, 0.8445144316285058, 0.891901585346906, 0.8502390762752825, 0.8429415805005327, 0.8038121973409076, 0.8181996341033592, 0.8396703702791426, 0.8011255711455898, 0.7982876040256206, 0.7797625763310273, 0.8037193608011411, 0.7750240246478391, 0.7847620046610068, 0.7482775312049172, 0.7543300135592744, 0.7373133906400149, 0.7459695593925335, 0.7325559411512327, 0.7348669174521247, 0.711011884499098, 0.7247001720566365, 0.7167529817075862, 0.6914167736376535, 0.7058147925498585, 0.7090678492967399, 0.6932221202363742, 0.6790928706435408, 0.6715511326121569, 0.6632128489202684, 0.6550479483575318, 0.6382672749231605, 0.643384059157045, 0.6518526425733457, 0.6496456467561785, 0.6396363816685172, 0.6243684859282219, 0.6294160648518239, 0.6268524620982122, 0.6073787657820517, 0.6006031116283311, 0.6048366164968193, 0.5986716995983958, 0.6161900446824341, 0.6076604883902976, 0.6247964712902758, 0.6396124918639056, 0.8634665429655666, 0.8600209313316465, 0.8540416673242947, 0.8486580315310069, 0.8557903450668559, 0.8742204418712279, 0.8726896315386278, 0.8537166317679153, 0.8450104809543451, 0.8409226164879605, 0.8377319181966801, 0.8358503127493327, 0.843133529721562, 0.8496786502651986, 0.8571414872369849, 0.8608430579232892, 0.8639671040058002, 0.8538955908273301, 0.8539534356039323, 0.8541636587857792, 0.865676333264339, 0.8551721041629999, 0.8405315992071611, 0.8502189638359555, 0.8435940539719008, 0.8267211986193356, 0.8386437404866222, 0.8524548986318788, 0.8700593673461686, 0.8369870244773118, 0.8227742899909446, 0.8513910405682684, 0.8746220605499083, 0.8684492462488609, 0.8776212243127574, 0.8632047060067186, 0.8534807327206897, 0.841030231945668, 0.8405147491533876, 0.8460475403874165, 0.8541020276367094, 0.8527912130253007, 0.8633800518764363, 0.8707734340988642, 0.8649354244879944, 0.8747895855169379, 0.8753343114004227, 0.8771815630247227, 0.8475704281037649, 0.8460628258948871, 0.8503624555428222, 0.8478040454708079, 0.8494544154138536, 0.8598522432709284, 0.8717317806346571, 0.8763542224212424, 0.8868071210445677, 0.8770956568550381, 0.9213839144485779, 0.9022223399275697, 0.9043349447026628, 0.885560842407673, 0.8800218157207178, 0.9170358458314632, 0.9326729503462662, 0.9304013522750739, 0.9253084142710616, 0.923981829870269, 0.9484412315847736, 0.9478332548439478, 0.9451007115648568, 0.9503849677033877, 0.9407741048067239, 0.9616081244797985, 0.9560778675728979, 0.9741029890840162, 0.9578647702562173, 0.9579748812686798, 0.9935382056758716, 1.0442782317787886, 1.0790901299742002, 1.171911938431785, 1.2262254406308364, 1.3838193079695669, 1.5490048911024867, 1.666850767706741, 1.6768826968635346, 1.6410201639607613, 1.5727363019024627, 1.5853522020938506, 1.5303364430002244, 1.5340642143849983, 1.510412612056671, 1.473052529382884, 1.4784887860149662, 1.477804450820576, 1.438055446722383, 1.4588887544499742, 1.4502445416900303, 1.4983990750238607, 1.483070840494271, 1.5272146950060612, 1.5159799554756728, 1.481013027613488, 1.488872809146604, 1.4799271578405016, 1.4880083248962228, 1.4720365061491762, 1.4745633644315166, 1.4888092885484967, 1.4916646180625628, 1.5627355929388622, 1.6284490385437949, 1.5921480312654843, 1.6111453361355705, 1.5689812554660556, 1.5529963254064727, 1.6023613557387841, 1.5732398292326917, 1.5503871969847196, 1.554556854215706, 1.5260453913477428, 1.5229204319279255, 1.5043274963177573, 1.4989838192840903, 1.5830680506295873, 1.6824708879633195, 1.779868845002795, 1.8612372682288312, 1.9309162731854927, 1.9524161962111986, 1.9850387090854182, 1.9970679278967687, 2.0154497083717695, 2.0080835777018837, 1.888492893432189, 1.787296730082384, 1.7257518078727099, 1.6604669876423384, 1.599001255803534, 1.5580558030810556, 1.5800583346017014, 1.5565206284776767, 1.531166666748007, 1.5016274390218738, 1.4829633046829878, 1.4733726062548038, 1.4556708000449616, 1.4315277991297093, 1.4274118987738837, 1.4167883445609828, 1.405780502282574, 1.4150694353921496, 1.4072997194205434, 1.380543982960022, 1.385996729500451, 1.3818270119527065, 1.3732739560516498, 1.3676584441901098, 1.3635645963285796, 1.3436832572828594, 1.3293459925507813, 1.3088825191319966, 1.3007087366853136, 1.3054333140165284, 1.319894697232918463, 1.0332344229896888, 1.0313084800581924, 1.0344644696007022, 1.0569526957993107, 1.043763277333326, 1.0303034439286707, 1.041822791903078, 1.0349784903868777, 1.0273782131677134, 1.0291806070496232, 1.029288557097612, 1.0358953052653856, 1.0301992107107814, 1.0236805277640824, 1.013628208290719, 1.0035653458609175, 1.0138887905484486, 0.9918698991896948, 0.9891695001505818, 0.9777218815950082, 0.9759551761730438, 0.980006400250529, 0.9668508103362277, 0.9672052652364742, 0.9654768578281471, 0.9630888330888323, 0.9805472164213487, 0.9884012741965111, 0.975160385589525, 0.965430675825918463, 1.0332344229896888, 1.0313084800581924, 1.0344644696007022, 1.0569526957993107, 1.043763277333326, 1.0303034439286707, 1.041822791903078, 1.0349784903868777, 1.0273782131677134, 1.0291806070496232, 1.029288557097612, 1.0358953052653856, 1.0301992107107814, 1.0236805277640824, 1.013628208290719, 1.0035653458609175, 1.0138887905484486, 0.9918698991896948, 0.9891695001505818, 0.9777218815950082, 0.9759551761730438, 0.980006400250529, 0.9668508103362277, 0.9672052652364742, 0.9654768578281471, 0.9630888330888323, 0.9805472164213487, 0.9884012741965111, 0.975160385589525, 0.9654306758253608, 0.9531202913198373, 0.95040725961266, 0.9215562534704178, 0.9257132688013315, 0.9202586446945111]
    df.plot_points(data)

if __name__ == "__main__":
    plot()