function Ham_XYZ_nn(nqubit::Int, Jx, Jy, Jz)
    H=QubitsOperator()
    link=[(i,i+1) for i in 1:nqubit-1]
    for (key,jx,jy,jz) in zip(link,Jx,Jy,Jz)
        H+=QubitsTerm(key[1]=>"X",key[2]=>"X",coeff=jx)
        H+=QubitsTerm(key[1]=>"Y",key[2]=>"Y",coeff=jy)
        H+=QubitsTerm(key[1]=>"Z",key[2]=>"Z",coeff=jz)
    end
    return H
end

function Ham(nqubit::Int,ps1,ps2)
    H=QubitsOperator()
    link=[(i,i+1) for i in 1:nqubit-1]
    for (key,p1,p2) in zip(link,ps1,ps2)
        H+=QubitsTerm(key[1]=>"X",key[2]=>"X",coeff=p1[1])
        H+=QubitsTerm(key[1]=>"Y",key[2]=>"Y",coeff=p1[1])
        H+=QubitsTerm(key[1]=>"Z",key[2]=>"Z",coeff=p2[1])
    end
    return H
end

function Ham_fc(nqubit::Int, ps1, ps2, ps3)
    H = QubitsOperator()
    link = [(i, j) for i in 1:nqubit for j in i+1:nqubit]
    for (key, p1, p2, p3) in zip(link, ps1, ps2, ps3)
        H += QubitsTerm(key[1]=>"X", key[2]=>"X", coeff=p1[1])
        H += QubitsTerm(key[1]=>"Y", key[2]=>"Y", coeff=p2[1])
        H += QubitsTerm(key[1]=>"Z", key[2]=>"Z", coeff=p3[1])
    end
    return H
end
