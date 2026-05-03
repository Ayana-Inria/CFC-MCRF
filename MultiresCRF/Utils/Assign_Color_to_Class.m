function [ Class_RGB ] = Assign_Color_to_Class( label )

Class_RGB = label;
Class_RGB(:,:,1) = 233*(label==1) + 255*(label==2) + 0*(label==3) + 158*(label==4) + 249*(label==5) + 121*(label==6);
Class_RGB(:,:,2) = 199*(label==1) + 243*(label==2) + 128*(label==3) + 229*(label==4) + 198*(label==5) +  192*(label==6);
Class_RGB(:,:,3) = 255*(label==1) + 191*(label==2) + 64*(label==3) + 1*(label==4) + 171*(label==5) +  244*(label==6);

end


