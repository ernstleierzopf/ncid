import tensorflow as tf
import cipherTypeDetection.config as config
from cipherImplementations.cipher import OUTPUT_ALPHABET
from cipherImplementations.simpleSubstitution import SimpleSubstitution
import sys
from util.utils import map_text_into_numberspace
import copy
import math
import multiprocessing
import numpy as np
from py_mini_racer import py_mini_racer
sys.path.append("../")


js_functions = """var max_period = 15;

var cipher_symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ#0123456789";
var numb_symbols;
var cipher_values = new Array(9);
numb_symbols = cipher_symbols.length;

var logdi = new Array(
[4,7,8,7,4,6,7,5,7,3,6,8,7,9,3,7,3,9,8,9,6,7,6,5,7,4],
[7,4,2,0,8,1,1,1,6,3,0,7,2,1,7,1,0,6,5,3,7,1,2,0,6,0],
[8,2,5,2,7,3,2,8,7,2,7,6,2,1,8,2,2,6,4,7,6,1,3,0,4,0],
[7,6,5,6,8,6,5,5,8,4,3,6,6,5,7,5,3,6,7,7,6,5,6,0,6,2],
[9,7,8,8,8,7,6,6,7,4,5,8,7,9,7,7,5,9,9,8,5,7,7,6,7,3],
[7,4,5,3,7,6,4,4,7,2,2,6,5,3,8,4,0,7,5,7,6,2,4,0,5,0],
[7,5,5,4,7,5,5,7,7,3,2,6,5,5,7,5,2,7,6,6,6,3,5,0,5,1],
[8,5,4,4,9,4,3,4,8,3,1,5,5,4,8,4,2,6,5,7,6,2,5,0,5,0],
[7,5,8,7,7,7,7,4,4,2,5,8,7,9,7,6,4,7,8,8,4,7,3,5,0,5],
[5,0,0,0,4,0,0,0,3,0,0,0,0,0,5,0,0,0,0,0,6,0,0,0,0,0],
[5,4,3,2,7,4,2,4,6,2,2,4,3,6,5,3,1,3,6,5,3,0,4,0,5,0],
[8,5,5,7,8,5,4,4,8,2,5,8,5,4,8,5,2,4,6,6,6,5,5,0,7,1],
[8,6,4,3,8,4,2,4,7,1,0,4,6,4,7,6,1,3,6,5,6,1,4,0,6,0],
[8,6,7,8,8,6,9,6,8,4,6,6,5,6,8,5,3,5,8,9,6,5,6,3,6,2],
[6,6,7,7,6,8,6,6,6,3,6,7,8,9,7,7,3,9,7,8,9,6,8,4,5,3],
[7,3,3,3,7,3,2,6,7,2,1,7,3,2,7,6,0,7,6,6,6,0,3,0,4,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,0,0,0,0,0],
[8,6,6,7,9,6,6,5,8,3,6,6,6,6,8,6,3,6,8,8,6,5,6,0,7,1],
[8,6,7,6,8,6,5,7,8,4,6,6,6,6,8,7,4,5,8,9,7,4,7,0,6,2],
[8,6,6,5,8,6,5,9,8,3,3,6,6,5,9,6,2,7,8,8,7,4,7,0,7,2],
[6,6,7,6,6,4,6,4,6,2,3,7,7,8,5,6,0,8,8,8,3,3,4,3,4,3],
[6,1,0,0,8,0,0,0,7,0,0,0,0,0,5,0,0,0,1,0,2,1,0,0,3,0],
[7,3,3,4,7,3,2,8,7,2,2,4,4,6,7,3,0,5,5,5,2,1,4,0,3,1],
[4,1,4,2,4,2,0,3,5,1,0,1,1,0,3,5,0,1,2,5,2,0,2,2,3,0],
[6,6,6,6,6,6,5,5,6,3,3,5,6,5,8,6,3,5,7,6,4,3,6,2,4,2],
[4,0,0,0,5,0,0,0,3,0,0,2,0,0,3,0,0,0,1,0,2,0,0,0,4,4]);

var sdd = new Array(
[0,3,4,2,0,0,1,0,0,0,4,5,2,6,0,2,0,4,4,3,0,6,0,0,3,5],
[0,0,0,0,6,0,0,0,0,9,0,7,0,0,0,0,0,0,0,0,7,0,0,0,7,0],
[3,0,0,0,2,0,0,6,0,0,8,0,0,0,6,0,5,0,0,0,3,0,0,0,0,0],
[1,6,0,0,1,0,0,0,4,4,0,0,0,0,0,0,0,0,0,1,0,0,4,0,1,0],
[0,0,4,5,0,0,0,0,0,3,0,0,3,2,0,3,6,5,4,0,0,4,3,8,0,0],
[3,0,0,0,0,5,0,0,2,1,0,0,0,0,5,0,0,2,0,4,1,0,0,0,0,0],
[2,0,0,0,1,0,0,6,1,0,0,0,0,0,2,0,0,1,0,0,2,0,0,0,0,0],
[5,0,0,0,7,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,5,0,0,0,4,0,0,0,1,1,3,7,0,0,0,0,5,3,0,5,0,0,0,8],
[0,0,0,0,6,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,9,0,0,0,0,0],
[0,0,0,0,6,0,0,0,5,0,0,0,0,4,0,0,0,0,0,0,0,0,1,0,0,0],
[2,0,0,4,2,0,0,0,3,0,0,7,0,0,0,0,0,0,0,0,0,0,0,0,7,0],
[5,5,0,0,5,0,0,0,2,0,0,0,0,0,2,6,0,0,0,0,2,0,0,0,6,0],
[0,0,4,7,0,0,8,0,0,2,2,0,0,0,0,0,3,0,0,4,0,0,0,0,0,0],
[0,2,0,0,0,8,0,0,0,0,4,0,5,5,0,2,0,4,0,0,7,4,5,0,0,0],
[3,0,0,0,3,0,0,0,0,0,0,5,0,0,5,7,0,6,0,0,3,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,0,0,0,0],
[1,0,0,0,4,0,0,0,2,0,4,0,0,0,2,0,0,0,0,0,0,0,0,0,5,0],
[1,1,0,0,0,0,0,1,2,0,0,0,0,0,1,4,4,0,1,4,2,0,4,0,0,0],
[0,0,0,0,0,0,0,8,3,0,0,0,0,0,3,0,0,0,0,0,0,0,2,0,0,0],
[0,4,3,0,0,0,5,0,0,0,0,6,2,3,0,6,0,6,5,3,0,0,0,0,0,6],
[0,0,0,0,8,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[6,0,0,0,2,0,0,6,6,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0],
[3,0,7,0,1,0,0,0,2,0,0,0,0,0,0,9,0,0,0,5,0,0,0,6,0,0],
[1,6,2,0,0,2,0,0,0,6,0,0,2,0,6,2,1,0,2,1,0,0,6,0,0,0],
[2,0,0,0,8,0,0,0,0,6,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,9]);


function convert_string() {
    var num_code = new Array(),i,clen,n;

    code = document.puzzle.ciphertext.value;
    code = code.toUpperCase();
    code = code.replace(/Ø/g,'0');
    clen=0;
    for (i=0;i<code.length;i++) {
        n = cipher_symbols.indexOf(code.charAt(i))
        if ( n != -1){
            num_code[clen]=n;
            clen++;
        }
    }
    return num_code;
}

// vig family functions
function decode_let(ct, key, ciph_type){
        var j,k;
        var cp;

        switch(ciph_type) {
        case 2: // VIGENERE
        case 8: //VEAUTOKEY
                cp = (26+ct - key)%26;
                break;
        case 3: //VARIANT
        case 6: //VAUTOKEY
                cp = (ct+key)%26;
                break;
        case 4: //BEAUFORT
        case 7: // BAUTOKEY
                cp = (26+key - ct)%26;
                break;
        default: /* must be porta */
                key = Math.floor(key /2);
                cp = ct;
                if ( cp<13) {
                        cp += key;
                        if ( cp <13)
                                cp += 13;
                }
                else {
                        cp -= key;
                        if ( cp >12)
                                cp -= 13;
                }
        } /* end switch */
        return(cp);
} /* end decode_let */


function best_di(col,ciph_type,period,buffer){
/* return best log_di score for all possible digraph keys in column */
        var j,k,rows,ct;
        var best_score, score;
        var kl,kr,pl,pr, kl1,kr1;
        var cl,cr;

        best_score = 0;
        rows = Math.floor(buf_len / period);
        for (kl = 0;kl<26;kl++) for (kr = 0; kr < 26;kr++) {
                score = 0;
                ct = 0;
                kl1 = kl;
                kr1 = kr;
                for (j=0;j<rows;j++) {
                    if ( col+j*period+1>=buf_len)
                            break;
                    cl = buffer[col+j*period];
                    cr = buffer[col+1+j*period];
                    pl = decode_let(cl,kl1,ciph_type);
                    pr = decode_let(cr,kr1,ciph_type);
                    score += logdi[pl][pr];
                    ct++;
                    if ( ciph_type <= 9 // PAUTOKEY
                        && ciph_type >= 6 ){ //VAUTOKEY)
                            kl1 = pl;
                            kr1 = pr;
                    }
                }/* next j */
                score *= 100;
                score /= ct;
                if ( score > best_score)
                        best_score = score;
        } /* next kr,kl */
        return(best_score);
} /* end best_di */

function decode_sl(cl,cr,k, ciph_type) {
        var j,pl,pr

        if ( ciph_type == 1) { //BSLIDEFAIR
                pl = (26+k-cr)%26;
                pr = (26+k - cl) % 26;
        }
        else {
                pl = (26+cr-k) % 26;
                pr = (cl+k)%26;
        }
        return( [pl,pr]);
} /* end decode_sl */

function best_sldi(col, ciph_type,period,buffer){
/* return best log_di score for all possible single letter keys in column */
        var j,rows,ct, rowb, posn;
        var best_score, score;
        var k,pl,pr, kl1;
        var cl,cr;
        var result;

        best_score = 0;
        rows = Math.floor(buf_len / (2*period));
        rowb = 2*col;
        for (k = 0;k<26;k++) {
                score = 0;
                ct = 0;
        for (j=0;j<rows;j++) {
                        posn = j*period*2+rowb;
                        if ( posn+1 >= buf_len)
                                break;
                        cl = buffer[posn];
                        cr = buffer[posn+1];
                        result=decode_sl(cl,cr,k,ciph_type);
                        pl= result[0];
                        pr = result[1];
                        score += logdi[pl][pr];
                        ct++;
                }/* next j */
                score *= 100;
                score /= ct;
                if ( score > best_score)
                        best_score = score;
        } /* next k */
        return(best_score);
} /* end best_sldi */


/*
original_attributes = ['IC'0,'MIC'1,'MKA'2,'DIC'3,'EDI'4,'LR'5,'ROD'6,'LDI'7,'SDD'8,'Cipher_type'9,
'DIV_2'10, 'DIV_3'11, 'DIV_5'12, 'DIV_25'13, 'DIV_4_15'14, 'DIV_4_30'15,'PSQ'16,'HAS_L17','HAS_D'18,'HAS_J'19,'HAS_H'20,'DBL'21]
*/

function get_vig_values(dat) {
    var s,type_name,hi,n,i;
    var ciph_type, start_type;
    var best_score;
    var period,best_period;
    var attribute_group_scores;
    var group_index;

    // translate cipher type index to attribute_group_index
    // A_LDI 0, B_LDI 1, P_LDI 2, S_LDI 3, V_LDI 4
/* cipher types:
VIGENERE 2
VARIANT 3
BEAUFORT 4
VSLIDEFAIR 0
BSLIDEFAIR 1
VAUTOKEY 6
BAUTOKEY 7
VEAUTOKEY 8
PORTA 5
PAUTOKEY 9
*/

    var xlate_indices = [ 3,3,4,4,1,2,0,0,0,0];

    buf_len = dat.length;
    best_score = 0;

    var min_period = 3;
    if ( (buf_len%2) == 0) start_type = 0;
    else start_type = 2;
    attribute_group_scores = [0,0,0,0,0];
    for (ciph_type = start_type; ciph_type<=9;ciph_type++) {
        group_index = xlate_indices[ciph_type];
        for (period = min_period; period <= max_period; period++) {
            sum = 0;
            for (col = 0; col <period;col++)
                if ( ciph_type > 1) //BSLIDEFAIR
                    sum += best_di(col,ciph_type,period,dat);
                else
                    sum += best_sldi(col,ciph_type,period,dat);
                sum /= period;
                n = Math.floor(sum) / 1000;
                if (n > attribute_group_scores[group_index]) {
                    attribute_group_scores[group_index] = n;
                }
         } /* next period */
    } /* next ciphertype */
    return(attribute_group_scores);
}


function decode_pair(k,c1, c2) {
        var t_flag,b_flag,t_index,b_index;
        var rvalue,sum;

        if (c1<13) t_flag = 0;
        else t_flag = 2;
        if ( c2 % 2 ) b_flag = 1;
        else b_flag = 0;
        rvalue = [0,0,0];
        sum = t_flag+b_flag;
        if ( sum == 2)
            if (c1-13 != (c2 >> 1)) // c1,c2 not verticaly aligned
                rvalue = [1, (c2 >> 1)+13,(c1-13) << 1]
        if ( sum == 3)
            if (c1-13 != (c2>>1))// c2, c2 not vertically aligned
                rvalue = [1,(c2>>1)+13,( (c1-13)<<1 )+1 ]
        return(rvalue);
} /* end decode_pair */


function calc_portax_logdi(nc){
    var s, count,score,hi,j,k,result
    var big_step;
    var best_score;
    var period,best_period;
    var c1,c2,c3,c4;

    if (nc.length&1 != 0) {//odd number of letters
        return 0
    }
    var buf_len = nc.length;

    best_score = 0;
    for (period = 3; period <= max_period; period++) {
        /* do encryption/decryption */
        big_step = 2*period;
        count = 0;
        score = 0;
        for (j=0;j<buf_len;j=j+big_step)
                for (k=0;k<period;k++) {
                        c1 = nc[j+k];
                        c2 = nc[j+k+period];
                        if (j+k+period >= buf_len) break;
                        result = decode_pair(k,c1,c2)
                        if (result[0]==1 ) {
                                c3 = result[1];
                                c4 = result[2];
                                /* plaintext independent of key values*/
                                score += logdi[c3][c4];
                                count++;
                        }
        } /* next k,j */
        /* skip testing of remainder, probably won't be crucial  */
        score *= 100;
        score /= count;

        if ( score > best_score) {
                best_score = score;
                best_period = period;
        }
    } /* period */
    best_score = Math.floor(best_score);
    return best_score / 1000;
}


var columnar_calcs = function(){
    // put pseudo-global variables in this closure
    var code = [];
    var numb_long_cols, numb_short_cols;
    var min_start = [];
    var max_start=[];
    var max_diff=[];
    var offset=[];
    var test_len;
    var col_array=[];
    var cols_in_use=[];
    var best_col_array=[];
    var diff_array=[];
    var next_col, next_dif;

    var key_len, numb_rows;

    var col_pos = [];

    function get_best_di(col){
        var i,j,k;
        var max,sum;
        var index,dif,long_corr,short_corr;
        max = 0;

        for (j= col;j<key_len;j++) {
        long_corr = short_corr = 0;
        if ( col>=numb_long_cols && col_array[j] >= numb_short_cols)
            short_corr = 1;
        else if ( col<numb_long_cols && col_array[j] >= numb_long_cols)
            long_corr=1;
        for (dif = short_corr;dif<=max_diff[ col_array[j] ] - long_corr ;dif++) {
            sum = 0;
            for (k=0;k<numb_rows;k++) {
                    try {
                        sum += sdd[(code[col_pos[ col_array[col-1]]+k+diff_array[col-1]])%sdd.length] [code[col_pos[col_array[j]]+k+dif] ];
                    } catch(e) {}
            }
            if ( sum > max) {
                max = sum;
                next_col = j;
                next_dif = dif;
            }
        }
        }
        return(max);
    }

    var do_col_calc = function(dat){ // return this function which can use the pseudo-global variables
        var str, alpha,out_str,c,n,i,ct,sum,c1,c2;
        var j,best_score, current_dif,index,t0,score,tn,swap;
        var normal_score,best_key_len;

        /*
        alpha="ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        out_str="";
        code = [];
        code_len = 0;
        str = document.getElementById('input_area').value;
        str = str.toUpperCase();
        for (i=0;i<str.length;i++){
            c = str.charAt(i);
            n = alpha.indexOf(c);
            if ( n>=0) {
                    code[code_len++] = n
            }
        }
        if ( code_len == 0){
            alert("No letters entered!");
            return;
        }
        */
        code = dat; // put in clsoure space so won't have to keep passing the entire array to get_best_di
        var code_len = code.length;
        for (i=0;i<max_period;i++) cols_in_use[i] = 0;
        best_score = 0;
        for (key_len = 4;key_len <= max_period;key_len++){
            numb_long_cols = code_len % key_len;
            numb_short_cols = key_len - numb_long_cols;
            /* transpose into columns */
            numb_rows = Math.floor(code_len / key_len);
            /* get min_start,max_start,max_diff*/
            min_start[0] = 0;
            n = 0;
            for (j=1;j<key_len;j++) {
                if ( n<numb_short_cols) {
                    min_start[j] = min_start[j-1]+numb_rows;
                    n++;
                }
                else {
                    min_start[j] = min_start[j-1]+numb_rows+1;
                }
            }
            max_start[0]= max_diff[0] = 0;
            n = 0;
            for (j=1;j<key_len;j++) {
                if ( n<numb_long_cols) {
                    max_start[j] = max_start[j-1]+numb_rows+1;
                    n++;
                }
                else {
                    max_start[j] = max_start[j-1]+numb_rows;
                }
                max_diff[j] = max_start[j]-min_start[j];
            }

            /* set column pointers to minimum for each column*/
            for (j=0;j<key_len;j++) {
                col_pos[j] = min_start[j];
            }
            /* try all possible digraphs */
            for (t0=0;t0<key_len;t0++) {
                col_array[0] = t0;
                cols_in_use[t0] = 1;
                if (0<numb_long_cols && t0 >= numb_long_cols)
                    long_corr=1;
                else long_corr=0;
                for ( current_dif=0;current_dif <= max_diff[t0] - long_corr ;current_dif++) {
                    diff_array[0] = current_dif;
                    index = 1;
                    for (j=0;j<key_len;j++)
                        if ( !cols_in_use[j])
                            col_array[index++] = j;
                    score = 0;
                    for (j=1;j<key_len;j++) {
                        tn = get_best_di( j);/* also sets next_col and next dif */
                        score += tn;
                        swap = col_array[next_col];
                        col_array[next_col] = col_array[j];
                        col_array[j] = swap;
                        diff_array[j] = next_dif;
                    }
                    score = Math.floor(100*score/(numb_rows*(key_len-1)));
                    if ( score > best_score ) {
                        best_score = score;
                        best_key_len = key_len;
                    }
                } /* next current_dif*/
                cols_in_use[t0] = 0;
            } // next t0
        } //next key_len
        normal_score = best_score;
        out_str = "CDD "+best_score;
        return(best_score / 1000);
    } // end do_cal_calc function
    return (do_col_calc); // return this function which can access pseudo_global variables
} // end columnar_calcs closure function

function get_cdd(dat){
    var s;

    do_col_calc = columnar_calcs(); // closure to isolate 'pseudo_global' variables
    s= do_col_calc(dat);
    return(s);
}


var swagman_calcs = function(){
// T has compressed binary Single letter - Trigraph Discrepancy values
var T="sP5D4475HAAPRphXWR=><I@A42p`E1"
T=T+"N71rHAR2ApH2`M8BAiEW75A@uiAYU1"
T=T+"r`NpS1IDR51@8D2@p5pI4PpTOO>D4p"
T=T+"18;`QR6@p@Xonj`1p44s@4Xq6cQ822"
T=T+"BrPq41r4262s4pRp4p2p@4111A44@p"
T=T+"T8>4r1t1@@@458A1665pP`4v4tb7lX"
T=T+":r4XS^V6s1p44p1:pPZnJ^q4r8q1rX"
T=T+"bXNt1p8p1Pu@41118PX3xA4544Ap@@"
T=T+"s8PU1v1p@rH1?22r1BRQy1p24hN_^?"
T=T+"q1r2t@;@@2s@t911s4A@@P@@Qx@q11"
T=T+"p4u2p:Cv@u2PQP2q@tjeW3E@82B6Y8"
T=T+"@2@rB@p@BP8p<ApCAqR`NP1d5TT1hG"
T=T+"S8HNiYApD4t@<@CIslG`@T4AIAQp2Q"
T=T+"p421@6YI@H@435AqP68qA444S2p44w"
T=T+"41:pPSH8RAT44q8q1rXPRC1r@1P8pA"
T=T+"Ps@r115Hc[c3q@t@q44@p@s9r1v1q4"
T=T+"4p\\\\p6::sha[[1w@T24882Rr1p1424s"
T=T+"P^hH:s@r@p8u4A@@rPx@4p11p4u@82"
T=T+"I}2PQPr@xTA8N64Yq2Bp8q@rQp14{4"
T=T+"p4r2s1q44qPp8s1t14p@65HAig_g1p"
T=T+"@uhcWc31q`4H8Xr4p6VT`p@q@42TTA"
T=T+":BTQD8RhT61UE8p8KTb<wH7XK@oU\\\\5"
T=T+"WCD6D4511p988qdD2sg1\\\\F4A9FD1QJ"
T=T+"9p1@p\\\\CH26NliaAp2P<p2:Hp1paPp1"
T=T+"p1p@@tAT2p887Rq81r2p@{@p22@p8q"
T=T+"144p4A@@1qXs4t4A1114p61p8Dr1u1"
T=T+"v;`S2r@PT@`4s4A@@qXq22p82p@rPw"
T=T+"22t4@q4pB@t14p4pPdZ1q1t41@@@p1"
T=T+"p1sPp@t8v@qH88s`O]neNAAAHE2Q1p"
T=T+":pPX4H6hp6pQ48p4s7B1J2s1P88AT4"
T=T+"4v1pT:8;PEeF6s1p14p@p@qPPpV9]Y"
T=T+"v1tHp62sPl_RQr@tAT2pH<7R;q1p8p"
T=T+"2p@sZ8hHs@q6@p81q444p@@2pS@i4w"
T=T+"@4p111Dp54@p@2aZIv@uCPS1sPonjn"
T=T+"y8qJcYK2p@r1u\\\\?o^?sLTV71p2y`dW"
T=T+"><x1p@xX_f>4u4tPp8y@6q@meVCU5:"
T=T+"p4Q@2RhaWSU7:64C4`Dr[T7:>41P8p"
T=T+"oQXpU7F6D64A@pQqT7fF6sn38pTE;>"
T=T+"4GThIq@p41rFph@sl?o>=p1pPl3Z11"
T=T+"w@P2ph97Rq81r2p@s18:Ps@q2@48qA"
T=T+"4441@rW8Z@1444t4p11I4251q42Olk"
T=T+"lx41p83PSp2q@p:T@@s4r@4XqjRS91"
T=T+"p@rPq4t2u4pTp4p2@@s1444q28x41p"
T=T+"@@FU@Aq6Pz4q1q`ph:th_VdMN@H@Aq"
T=T+"1Pp:pPkfb>hAV3U7H44qPpep84yPqP"
T=T+"wVZ?KU7nN6s1p44uP@8^;=S|8p6ulS"
T=T+"YA5p@@tAPRphl?f>q1r2p@sglhOs@p"
T=T+"J2@p9p1p4T44A@qPHZp1p14s@4111A"
T=T+"44pD@8pb[JIv@t8;PQPr@8olZ^r44A"
T=T+"p@p4Xq6cQ86@BrPq41qh6_>5s4pRp4"
T=T+"p2@r1pA44D8488p4@p1s41@@@FeIA1"
T=T+"4p4Lpb8v4qAp@`pXXt@p2p@NmkmD4P"
T=T+"Q118qPqPh@VA5A82p1qHslOoN51p1p"
T=T+"VP<pPADN4p1q^o[KQ3N6DomkM6q2T7"
T=T+"9FD11@9~p4q2u`3RAq@@t1p2pXK:>5"
T=T+"q1r2uO8h:s@s411@t@@pPCjH@x4q1p"
T=T+"4~v8p2PQu2:Hx@4X8pVkSI2pBrPpP4"
T=T+"1p2x4PVA4p2r11pA44@X8Z>4r1t1@@"
T=T+"@FUhAA42Uq@w4q@@p`7lXZ@q42p681"
T=T+"|omiN|Pp1Q~q@s2pP~xRz8PSP1w8PY"
T=T+"AX?_>GpQqAP64889Rq11rRp@pP81w@"
T=T+"p:4PNmj5hc_C5p@pP22s8t@EpY11q4"
T=T+"4@8BRs1s1@r11p2Pq4p8qbH8}B8s@r"
T=T+"PuX5:^1~q@rP~@s1s8I^85}Hq14p40"

var bstd = []

function construct_table(){
    var i,n,index,c,ze,x,j,mask;
    // read T and put it into the working binary trigraph table: bstd
    n = 26*26*26
    i = index = 0;

    ze = '0'.charCodeAt(0);
    while (i < n){
        c = T.charCodeAt(index);
        index +=1
        x = c-ze;
        if (x > 63){
            x -= 63;
            //for j in range(6*x):
            for (j=0;j<6*x;j++){
                bstd[i] = 0;
                i += 1;
            }
        }
        else{
            mask = 1;
            while (mask < 64){
                if (mask & x ){
                    bstd[i] = 1;
                }
                else bstd[i] = 0;
                i += 1;
                mask += mask;
                if (i >= n)  break;
            }
        }
    }
}

construct_table();


function next_per(str,le){
    /*
    get next permutation of array str of length le
    return 0 if finished, 1 otherwise.
    */
    if (le < 2) return (0);
    //find last element not in reverse alphabetic order
    var last = le-2;
    while (str[last] >= str[last+1]){
        if (last == 0) return(0);
        last -= 1;
    }
    // find first element that is larger than the element at last
    var fst = le-1;
    while (str[fst] <= str[last])
        fst -= 1;

    //swap these two
    var c = str[last];
    str[last] = str[fst];
    str[fst] = c;

    //put part of string at tail into ascending order
    if (str[last+1] != str[le-1] ){
        var i = 1;
        while (last+i < le -i){
            c = str[last+i];
            str[last+i] = str[le-i];
            str[le-i] = c;
            i += 1;
        }
    }
    return(1);
}

    function construct_row(row_order,swag_array,period,numb_columns){
        var i,c;
        var row = []
        var index = 0
        for (i=0;i<numb_columns;i++){
            c = swag_array[ row_order[index] ][i]
            row[i] = c;
            index += 1;
            if (index == period ) index = 0;
        }
        return(row);
    }

    function score_row(row){
        var score = 0
        for (var i=0;i<row.length-2;i++)
            score += bstd[row[i]+26*row[i+1]+26*26*row[i+2]];
        return score;
    }


    function swag_test(code,period){
        var i,j,index,c;
        /*
        test code digits for swagman of given period, return binary std score
        and best scoring line
        */
        var numb_columns = code.length/period; // should always be integer
        var row_order = [];
        for (i=0;i<period;i++)
            row_order[i] = i;
        var swag_array = [];
        for (i=0;i<period;i++)
            swag_array[i] = []
        index = i = 0;
        //for c in code:
        for (j=0;j<code.length;j++){
            c = code[j];
            swag_array[i][index] = c;
            i += 1;
            if (i == period ){
                index += 1;
                i = 0;
            }
        }
        var row = construct_row(row_order,swag_array,period,numb_columns)
        var score = score_row(row)
        var best_score = score
        var best_row = row
        while (next_per(row_order,row_order.length) != 0){
            row = construct_row(row_order,swag_array,period,numb_columns)
            score = score_row(row)
            if (score > best_score){
                best_score = score;
                best_row = row;
            }
        }
        var std_score = Math.floor(100*best_score / (numb_columns-2));
        var alpha="ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        var l1 = '';
        for (i=0;i<best_row.length;i++)
            l1 += alpha.charAt(best_row[i]);
        return ([std_score,l1]);
    }
    var do_swag_calc = function(code){ // return this function which can use the pseudo-global variables
        var str, alpha,out_str,c,n,i,ct,sum,c1,c2;
        //var code_len,code;
        var code_len;
        var period, best_score, best_line,result,best_period;

        /*
        alpha="ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        out_str="";
        code = [];
        code_len = 0;
        str = document.getElementById('input_area').value;
        str = str.toUpperCase();
        for (i=0;i<str.length;i++){
            c = str.charAt(i);
            n = alpha.indexOf(c);
            if ( n>=0) {
                    code[code_len++] = n
            }
        }
        if ( code_len == 0){
            alert("No letters entered!");
            return;
        }
        */
        //out_str = 'testng'
        best_score = 0;
        best_line = '';
        for (period = 4;period <= 8;period++){
            if ( code.length % period != 0) continue;
            if (3*period*period > code.length) break; // not enough code blocks
            result = swag_test(code,period);
            if (result[0] > best_score){
                best_score = result[0];
                best_line = result[1];
                best_period = period;
             }
        }
        /*
        out_str += 'Best score is '+best_score+' for period '+best_period+' best line is: '+best_line.toLowerCase();
        document.getElementById('output_area').value = out_str;
        */
        return(best_score / 100);

    } // end do_cal_calc function
    return (do_swag_calc); // return this function which can access pseudo_global variables
} // end swagman_calcs closure function


function get_sstd(dat){
    var s;

    do_swag_calc = swagman_calcs(); // closure to isolate 'pseudo_global' variables
    s= do_swag_calc(dat);
    return(s);
}"""
ctx = py_mini_racer.MiniRacer()
ctx.eval(js_functions)


global_english_frequencies = [
    0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015, 0.06094, 0.06966, 0.00153, 0.00772, 0.04025, 0.02406, 0.06749, 0.07507,
    0.01929, 0.00095, 0.05987, 0.06327, 0.09056, 0.02758, 0.00978, 0.0236, 0.0015, 0.01974, 0.00074]
global_english_digraph_frequencies = [0.00039803239807310176, 0.002029445518441609, 0.004140646019086559, 0.0034405166367435383, 0.00018870001483253996, 0.0013187785199617543, 0.0020372352972668983, 0.0005146170623011168, 0.003231846814847664, 0.0002012572289530235, 0.0011880571323692015, 0.008836830184180957, 0.0036946134220110184, 0.016136243079947415, 0.00017334501113159254, 0.001978181771203139, 7.286278455427354e-05, 0.009794636726918318, 0.008735606073905992, 0.011164000013278053, 0.001129515154540852, 0.00191689172480274, 0.0009063006657509358, 0.00015282295398409983, 0.0026649109948876704, 0.00017769108978803644, 0.002050693502311955, 0.00015937502659062186, 7.409124035287036e-05, 3.2781638999001434e-05, 0.004502292583201862, 1.7425941516541255e-05, 9.369750590351755e-06, 3.572720404168359e-05, 0.0010074775988830336, 6.535606858619135e-05, 6.242414791326018e-06, 0.0016051893354886344, 4.4357383539431316e-05, 2.4542521013947084e-05, 0.001889952188662201, 2.663219093038549e-05, 1.2749391599518517e-06, 0.0010686732909976969, 0.0003260014575526296, 9.904332371985113e-05, 0.0018762791426086922, 2.776999261131477e-05, 3.2420178830852556e-05, 6.986379833510873e-07, 0.0012099720715338155, 1.880610420592864e-06, 0.0046091962202008, 6.892788707439312e-05, 0.0006999080660404499, 8.289185884225322e-05, 0.004579794915992478, 6.189224875347616e-05, 4.199459496746903e-05, 0.004655909917018074, 0.0019532456448109517, 9.603323699648213e-06, 0.0016662529778553687, 0.0013143006228178858, 6.612709110737392e-05, 4.3487612782932326e-05, 0.0061832354595479444, 8.855496607042317e-05, 3.643416740318782e-05, 0.0012752506678510819, 0.0003195113627612476, 0.00274939878246978, 0.0010647337683077315, 2.1790289752821203e-05, 5.949245850083325e-05, 1.2256806725457672e-06, 0.00026486635569008074, 1.3393220843361428e-05, 0.004066497426128634, 0.001412242915276984, 0.0008747384633908653, 0.0009253368741585971, 0.006250933272000211, 0.0009328766603787876, 0.0005647702965981599, 0.0010605063262899698, 0.005012339706678417, 0.0001848617842434377, 0.00012158382254847205, 0.0007055630791509709, 0.0008197965178322364, 0.0006569005500643486, 0.0030342122816937782, 0.0007273242300802561, 6.551933850219463e-05, 0.0013186194127348277, 0.0024120209269313875, 0.003644589924856862, 0.0013554897374490383, 0.0002864311664512544, 0.0011347522799201861, 6.339544203112663e-06, 0.0005129450488553611, 2.2672317316045648e-05, 0.010020473709826474, 0.0022521993363070515, 0.005960924043027139, 0.010787830752016612, 0.004277843394579735, 0.0030647166985073916, 0.001916331611861437, 0.0017481307593864685, 0.0037063924445346875, 0.00029069283502364555, 0.0005577168512184154, 0.005340324916836537, 0.004196289840275599, 0.011329747191802887, 0.003127610073983783, 0.0032432844968670544, 0.00033797242629482943, 0.017838136076634363, 0.013198141738779546, 0.007602122951633152, 0.0008496811564944489, 0.0024453511158464792, 0.0034171991026206243, 0.001306474536093429, 0.0017410081671159522, 0.00010764390187305436, 0.0019328588750584475, 0.00020539517315563884, 0.00036326191873751665, 0.00017298910121554578, 0.001972487675067399, 0.0014073401925868009, 0.00014748823667197044, 0.0003486492612552243, 0.002773699867517286, 6.240911598048367e-05, 5.293668572624318e-05, 0.000668536884856893, 0.00028937904409897906, 0.0001235768255741323, 0.004376321055106181, 0.00027747675972654266, 1.0985798994078137e-05, 0.0019285682989230244, 0.0004734864565775405, 0.0031673619045809974, 0.0007259035968026243, 5.658597648337001e-05, 0.00023262563501052922, 4.466334118655925e-06, 0.00018268677920092957, 7.212090085662698e-06, 0.00259931904058714, 0.00027389962224674304, 0.0003005325069586413, 0.00020346114155856332, 0.0033359380928543697, 0.000338863704278224, 0.0003395565607489687, 0.0022849460549699105, 0.0016426757381861775, 4.092085244621809e-05, 3.788740841191944e-05, 0.0005959090609749415, 0.00027254304812878956, 0.0005611371940763308, 0.0018937247412680951, 0.0002810286898113786, 1.3817815129171621e-05, 0.0016165023680962318, 0.0009066972775157312, 0.0016992998726527494, 0.0008714890220455935, 4.319414320303411e-05, 0.00036261438932560564, 3.4175677318644052e-06, 0.0002265899671100062, 6.594162018296227e-06, 0.008318866088601775, 0.0002344990763554902, 0.0003332595684786388, 0.0001916582992029561, 0.0232854497343354, 0.000192936938530976, 9.93511314510131e-05, 0.0003075760081367029, 0.0063585866555539395, 4.391775732084462e-05, 4.865374118746061e-05, 0.00027045175938882136, 0.00031289569351605577, 0.0003200548249462443, 0.004562544501198666, 0.00022632286122759292, 2.3413044711170948e-05, 0.0008887343491083124, 0.0005693693742462576, 0.001931383895562316, 0.0006410148035061384, 4.568296874981478e-05, 0.00032451005856069603, 1.7404665550149893e-06, 0.00033450698763858445, 8.571901850675737e-06, 0.0023130703386737422, 0.0006009174697155686, 0.004964795784650871, 0.0029825174648753787, 0.0028920388739305716, 0.0013275310362662524, 0.0022040453490692834, 0.0001412268585192956, 0.00014040380238465591, 5.067565177615262e-05, 0.0005978370797989063, 0.004134382790849851, 0.00243850834878611, 0.020275533912479046, 0.0049050722969062885, 0.0007744037810152603, 6.744365715809148e-05, 0.002701435585148022, 0.00863757543993427, 0.008773684503494425, 0.0001333640013746624, 0.002111230795771007, 0.00021323582929186368, 0.00020336123702072563, 2.274701445891966e-05, 0.0004314863113579693, 0.00039609443504745395, 4.481828572440937e-06, 5.728322690369557e-06, 5.065298824673574e-06, 0.00034396484848105694, 2.923132773769528e-06, 2.7804450426448604e-06, 4.847220169162129e-06, 8.269343732960336e-05, 3.7198252109242765e-06, 3.2300154629144775e-06, 2.8095838661808537e-06, 5.1658971440240275e-06, 3.3421768074776277e-06, 0.0006293396169488794, 7.983112606845262e-06, 1.6697008407132905e-07, 1.8609764037817064e-05, 9.094550590289593e-06, 4.7195643708139654e-06, 0.0006763941917494241, 2.064000000466221e-06, 3.719362689915769e-06, 1.72751596677677e-07, 1.3235038658451746e-06, 6.611737816619525e-07, 0.0006551697964505124, 0.00010588493447769906, 9.713334321521803e-05, 6.428625749351273e-05, 0.0024630793148420803, 0.00012426598187680897, 4.8394960683200476e-05, 0.00015034129751295103, 0.0013446311317323, 1.7764506894768992e-05, 2.747629177091229e-05, 0.00019571784609462937, 0.00011230403229427506, 0.0004402820733767629, 0.00040655619773889267, 8.687370220449718e-05, 3.2156773116507347e-06, 0.00011725370086682168, 0.000746354656975311, 0.00033393669923509426, 0.0001171607341441116, 1.692456874331876e-05, 0.00016642269045776, 1.175497143122667e-06, 0.000127955511961676, 2.588267563609854e-06, 0.005360229277177168, 0.0005697548855068488, 0.0005383890233148899, 0.0023693977659133563, 0.007026448490998915, 0.0006249866004773079, 0.00021425638189713622, 0.00029463698292369613, 0.005386327487603231, 5.0498506229894114e-05, 0.00026923024140535217, 0.005697536135740754, 0.0005125921453258696, 0.00017398097751829082, 0.003606810515100429, 0.0005883167786202853, 1.7841285382181293e-05, 0.0003480683348685385, 0.0020062894041506644, 0.0015765659916166226, 0.0010182261245997473, 0.00028636687603107177, 0.000424781838079144, 3.576906219295355e-06, 0.0031779890185329777, 1.3254464540809076e-05, 0.005048041703325137, 0.0009532012210556475, 0.0002547859415701567, 0.00011126544136967071, 0.006299011868313592, 0.0001653713802054217, 6.59400013594325e-05, 0.00016439476709595742, 0.002814196125677694, 3.104972908264384e-05, 2.5679397652859347e-05, 0.00011066462657961904, 0.0008627191612032764, 0.00012913517179387524, 0.0029950011381555093, 0.0018119658276361818, 4.939261849855188e-06, 0.00015277508305971927, 0.0009072014254150048, 0.0007067196129327448, 0.0008685760647340111, 3.1524044376868626e-05, 0.00021683470525906318, 3.295462185618336e-06, 0.0004507565553959356, 4.225360673223342e-06, 0.005445612274171245, 0.0008331603685915576, 0.003518541387013264, 0.010682918499219806, 0.006320736942604214, 0.0011448165058048124, 0.008919108277644921, 0.0009054796909608344, 0.004035982371331825, 0.00031052157317938505, 0.0007037719665455243, 0.0008540415733021567, 0.0008780794838958217, 0.0011981373152286214, 0.004369461637289505, 0.0006864103154491656, 5.0281121355895434e-05, 0.0005535405177720939, 0.004927333664306275, 0.011725158252060271, 0.0008632034206991841, 0.0005075090394423684, 0.0009749866543378793, 1.7308461180380262e-05, 0.001004431435521001, 6.162190522400334e-05, 0.001515732453451621, 0.0014367086578035188, 0.0017684379755255094, 0.001759941927120229, 0.0006050487073635606, 0.0070629048594105116, 0.0009627666180325981, 0.0007526740815145536, 0.0012341485071695286, 0.00015288215667318884, 0.0007857237514379853, 0.003174395230296872, 0.004871769859251708, 0.013162249877258834, 0.002351654766245483, 0.0024188588375211673, 2.837034488035794e-05, 0.010574430727766728, 0.0031442790998699012, 0.004645572109956916, 0.007195042486331116, 0.0016997679439133594, 0.0033788151779060717, 0.00015033736608437872, 0.0004470015785883647, 5.285597581025856e-05, 0.0027910157290338025, 8.541282959912518e-05, 9.257542993687754e-05, 6.317158186300885e-05, 0.003601493373586623, 9.670574254285253e-05, 4.882672404464254e-05, 0.0006533904781307827, 0.0012856257078534253, 1.087918790161708e-05, 1.919855328164754e-05, 0.002269180332613408, 0.0002153555630738551, 3.044428908250708e-05, 0.0027560551535637206, 0.0011270233226075158, 4.303064202652658e-06, 0.003050599401025211, 0.0005497145439897171, 0.0008816748909554573, 0.0008922372519662465, 1.1124786557134741e-05, 0.00012266311532182507, 1.5758090759861996e-06, 9.161315497867699e-05, 2.242533109750246e-06, 1.7003891096277855e-05, 6.315030589661748e-06, 2.466855798876547e-06, 2.0068786559155033e-06, 1.3921882356085885e-06, 2.030004706340895e-06, 5.936457144198084e-07, 2.8382601687083397e-06, 1.6971514625682306e-05, 3.103515967087584e-07, 4.678400001056768e-07, 2.220794622350378e-06, 2.8479731098870043e-06, 8.806400001989211e-07, 2.172461176961309e-06, 1.401901176787253e-06, 5.779200001305419e-07, 1.381781512917162e-06, 4.821087732181436e-06, 3.911540168950775e-06, 0.0009642230966883892, 9.74069243917504e-07, 8.017570421979094e-06, 1.7691428575424752e-07, 1.053854117885106e-06, 6.475294119109714e-08, 0.006624590581664445, 0.0007738466744605126, 0.0016128253260785945, 0.0020872733638328225, 0.014089222456964019, 0.0007946647450534503, 0.0010744219646124408, 0.0006865444465416329, 0.006390801475057014, 0.00011982924910269757, 0.001038683428806049, 0.0011108010920156163, 0.0017062374565198627, 0.0016337710524698804, 0.006759922609930308, 0.0008298061662378588, 3.6292404714080165e-05, 0.0013635609603080043, 0.004911339225311066, 0.004961939023641823, 0.0012327472997742542, 0.0006226561883759411, 0.0007742613245446399, 8.93914353143096e-06, 0.002032441960795227, 2.6232341518530465e-05, 0.0069563462630839205, 0.0012843477623069182, 0.0024977605276230237, 0.0008885606724696177, 0.00729216912299171, 0.0014046751465357787, 0.0004726432807790307, 0.0038789618079350125, 0.005957002558656506, 0.00016290961213763874, 0.0005369609897011219, 0.0011482111787467556, 0.001290608215417576, 0.00096157886408275, 0.005527965758559595, 0.00244456829903958, 0.00018508841953760653, 0.0008126050099314522, 0.004374453395273826, 0.012492322191729358, 0.0023197752744735763, 0.00020399095937380905, 0.0020057764683522293, 1.1788504204343487e-05, 0.0005120731967543238, 1.846383865963284e-05, 0.006046905542206225, 0.000882364972300151, 0.0012018185199353351, 0.0005426564733998874, 0.009781351042209434, 0.0007789899080751198, 0.00035383897823118647, 0.027056980400061274, 0.009918454525937884, 0.00012938400809645246, 0.00014114591734280674, 0.0012495321871730035, 0.0008695073507846416, 0.00041213373858048867, 0.010664621630644244, 0.0007100684962948457, 3.67960900923452e-05, 0.0036588247026752034, 0.004376031979475863, 0.004478931340843644, 0.0019605097685100716, 0.00016145452104487308, 0.002060589833070493, 6.511370757773325e-06, 0.0018521464152082832, 6.475456001462691e-05, 0.0010614850207439723, 0.0006916696418369083, 0.0013279868507201369, 0.0008093042287542361, 0.0011396140695011162, 0.00016231989785179125, 0.001117525916218816, 7.847617077402891e-05, 0.0008051292828709401, 2.0389776139059473e-05, 0.0001190697896067277, 0.002352721339691102, 0.0010151688607335104, 0.0035238779544094273, 0.00015029758927764705, 0.0012272874705293235, 5.408258152482134e-06, 0.0040104542180487476, 0.0036306402912402655, 0.003500629335916781, 1.4579355969679774e-05, 4.9039021187547636e-05, 8.157298018649313e-05, 3.348975866302693e-05, 0.0001230213378429144, 3.5553064881980387e-05, 0.0009507986556769535, 6.750956640180384e-06, 1.3649920003083275e-05, 1.9798443029682204e-05, 0.006780783001195525, 6.496107564492566e-06, 5.916800001336501e-06, 6.984761009981095e-06, 0.0021692320865404112, 2.6437700846307946e-06, 2.6523267232881895e-06, 1.1339165044578123e-05, 8.099667900989235e-06, 7.650560001728127e-06, 0.0005210974441513202, 1.4471588574697448e-05, 3.4411563032983046e-07, 2.229721277814579e-05, 4.719865009469496e-05, 1.4549060843622511e-05, 1.9155307567352055e-05, 5.163815799485743e-06, 1.0547329078012708e-05, 7.381835295785074e-07, 5.390266085251179e-05, 6.08908907700567e-07, 0.0038941479914678546, 9.130627228953204e-05, 0.00010369582254443145, 0.00010005393212344076, 0.0030491965748064066, 7.775278791672264e-05, 3.2351031940080635e-05, 0.002741109712215807, 0.0035181702139039364, 2.299538824048837e-05, 3.4449489755680694e-05, 0.0001521189970091509, 0.00011694543061465121, 0.0008440118052326642, 0.002106007777282433, 7.440714220168121e-05, 3.7568268916049035e-06, 0.0002836999798960156, 0.00045991400884338226, 0.00030093767536209414, 4.1831325051465765e-05, 1.4784484036953e-05, 0.00015601064877473586, 1.08183663889983e-06, 0.00012803668439866913, 1.221888000276003e-05, 0.00020909372240017176, 2.1747969080542734e-05, 0.00016141867566671372, 1.3898987566164746e-05, 0.00015123211297533713, 2.6139606056324644e-05, 9.085993951632197e-06, 3.852776874819854e-05, 0.00023698096408714326, 2.458067899714898e-06, 3.156937143570239e-06, 1.3779657145969725e-05, 2.9483864208340556e-05, 8.0326023547556e-06, 4.88359744648127e-05, 0.00042568028513817046, 1.2525068910392216e-06, 2.082408336604833e-05, 3.5694365050079535e-05, 0.0003491961923477848, 3.4118555974093337e-05, 7.1961331108691766e-06, 2.7594465888586045e-05, 8.106143195108345e-06, 2.1814572105767862e-05, 4.81484369856658e-07, 0.001674221521050446, 0.0006236600902249075, 0.000723395114112982, 0.0004908127248167483, 0.0015030325516000127, 0.0005331118898683197, 0.00024261123232370915, 0.0005298809493633882, 0.00103170259922464, 8.757349649036954e-05, 9.064324842383606e-05, 0.00046574454867663205, 0.0005819145628205199, 0.00034357332444735506, 0.002101810399130224, 0.0005970829392945344, 2.034005513064488e-05, 0.00046759463271066337, 0.001743616554343432, 0.00155271794589695, 0.00016084445583465124, 9.516069111393209e-05, 0.0007814440445462624, 3.918709244582646e-06, 7.700350388293995e-05, 1.8103303533500983e-05, 0.00021486852845189634, 1.1713807061469471e-05, 9.490237313068047e-06, 7.6098581529794365e-06, 0.0003954256296691516, 6.627463530908792e-06, 6.0981082366715725e-06, 2.4892649417387516e-05, 0.0001489398588571723, 1.6574440339878327e-06, 5.6108423542085664e-06, 1.850985949997937e-05, 1.0645846052824878e-05, 5.605985883619234e-06, 9.805815397172944e-05, 7.027775463772324e-06, 1.3350668910578706e-06, 7.558749581539321e-06, 2.1968129080592464e-05, 1.317144201978192e-05, 2.625685513198138e-05, 3.316044370496935e-06, 1.5925754625446086e-05, 5.695946219774008e-07, 2.448378084586659e-05, 5.1172168078785784e-05]
global_logdi = [[4,7,8,7,4,6,7,5,7,3,6,8,7,9,3,7,3,9,8,9,6,7,6,5,7,4],[7,4,2,0,8,1,1,1,6,3,0,7,2,1,7,1,0,6,5,3,7,1,2,0,6,0],[8,2,5,2,7,3,2,8,7,2,7,6,2,1,8,2,2,6,4,7,6,1,3,0,4,0],[7,6,5,6,8,6,5,5,8,4,3,6,6,5,7,5,3,6,7,7,6,5,6,0,6,2],[9,7,8,8,8,7,6,6,7,4,5,8,7,9,7,7,5,9,9,8,5,7,7,6,7,3],[7,4,5,3,7,6,4,4,7,2,2,6,5,3,8,4,0,7,5,7,6,2,4,0,5,0],[7,5,5,4,7,5,5,7,7,3,2,6,5,5,7,5,2,7,6,6,6,3,5,0,5,1],[8,5,4,4,9,4,3,4,8,3,1,5,5,4,8,4,2,6,5,7,6,2,5,0,5,0],[7,5,8,7,7,7,7,4,4,2,5,8,7,9,7,6,4,7,8,8,4,7,3,5,0,5],[5,0,0,0,4,0,0,0,3,0,0,0,0,0,5,0,0,0,0,0,6,0,0,0,0,0],[5,4,3,2,7,4,2,4,6,2,2,4,3,6,5,3,1,3,6,5,3,0,4,0,5,0],[8,5,5,7,8,5,4,4,8,2,5,8,5,4,8,5,2,4,6,6,6,5,5,0,7,1],[8,6,4,3,8,4,2,4,7,1,0,4,6,4,7,6,1,3,6,5,6,1,4,0,6,0],[8,6,7,8,8,6,9,6,8,4,6,6,5,6,8,5,3,5,8,9,6,5,6,3,6,2],[6,6,7,7,6,8,6,6,6,3,6,7,8,9,7,7,3,9,7,8,9,6,8,4,5,3],[7,3,3,3,7,3,2,6,7,2,1,7,3,2,7,6,0,7,6,6,6,0,3,0,4,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,0,0,0,0,0],[8,6,6,7,9,6,6,5,8,3,6,6,6,6,8,6,3,6,8,8,6,5,6,0,7,1],[8,6,7,6,8,6,5,7,8,4,6,6,6,6,8,7,4,5,8,9,7,4,7,0,6,2],[8,6,6,5,8,6,5,9,8,3,3,6,6,5,9,6,2,7,8,8,7,4,7,0,7,2],[6,6,7,6,6,4,6,4,6,2,3,7,7,8,5,6,0,8,8,8,3,3,4,3,4,3],[6,1,0,0,8,0,0,0,7,0,0,0,0,0,5,0,0,0,1,0,2,1,0,0,3,0],[7,3,3,4,7,3,2,8,7,2,2,4,4,6,7,3,0,5,5,5,2,1,4,0,3,1],[4,1,4,2,4,2,0,3,5,1,0,1,1,0,3,5,0,1,2,5,2,0,2,2,3,0],[6,6,6,6,6,6,5,5,6,3,3,5,6,5,8,6,3,5,7,6,4,3,6,2,4,2],[4,0,0,0,5,0,0,0,3,0,0,2,0,0,3,0,0,0,1,0,2,0,0,0,4,4]]
global_sdd = [[0,3,4,2,0,0,1,0,0,0,4,5,2,6,0,2,0,4,4,3,0,6,0,0,3,5],[0,0,0,0,6,0,0,0,0,9,0,7,0,0,0,0,0,0,0,0,7,0,0,0,7,0],[3,0,0,0,2,0,0,6,0,0,8,0,0,0,6,0,5,0,0,0,3,0,0,0,0,0],[1,6,0,0,1,0,0,0,4,4,0,0,0,0,0,0,0,0,0,1,0,0,4,0,1,0],[0,0,4,5,0,0,0,0,0,3,0,0,3,2,0,3,6,5,4,0,0,4,3,8,0,0],[3,0,0,0,0,5,0,0,2,1,0,0,0,0,5,0,0,2,0,4,1,0,0,0,0,0],[2,0,0,0,1,0,0,6,1,0,0,0,0,0,2,0,0,1,0,0,2,0,0,0,0,0],[5,0,0,0,7,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,5,0,0,0,4,0,0,0,1,1,3,7,0,0,0,0,5,3,0,5,0,0,0,8],[0,0,0,0,6,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,9,0,0,0,0,0],[0,0,0,0,6,0,0,0,5,0,0,0,0,4,0,0,0,0,0,0,0,0,1,0,0,0],[2,0,0,4,2,0,0,0,3,0,0,7,0,0,0,0,0,0,0,0,0,0,0,0,7,0],[5,5,0,0,5,0,0,0,2,0,0,0,0,0,2,6,0,0,0,0,2,0,0,0,6,0],[0,0,4,7,0,0,8,0,0,2,2,0,0,0,0,0,3,0,0,4,0,0,0,0,0,0],[0,2,0,0,0,8,0,0,0,0,4,0,5,5,0,2,0,4,0,0,7,4,5,0,0,0],[3,0,0,0,3,0,0,0,0,0,0,5,0,0,5,7,0,6,0,0,3,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,0,0,0,0],[1,0,0,0,4,0,0,0,2,0,4,0,0,0,2,0,0,0,0,0,0,0,0,0,5,0],[1,1,0,0,0,0,0,1,2,0,0,0,0,0,1,4,4,0,1,4,2,0,4,0,0,0],[0,0,0,0,0,0,0,8,3,0,0,0,0,0,3,0,0,0,0,0,0,0,2,0,0,0],[0,4,3,0,0,0,5,0,0,0,0,6,2,3,0,6,0,6,5,3,0,0,0,0,0,6],[0,0,0,0,8,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[6,0,0,0,2,0,0,6,6,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0],[3,0,7,0,1,0,0,0,2,0,0,0,0,0,0,9,0,0,0,5,0,0,0,6,0,0],[1,6,2,0,0,2,0,0,0,6,0,0,2,0,6,2,1,0,2,1,0,0,6,0,0,0],[2,0,0,0,8,0,0,0,0,6,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,9]]


def calculate_frequencies(text, size, recursive=True):
    before = []
    if recursive is True and size > 1:
        before = calculate_frequencies(text, size-1, recursive)
    frequencies_size = int(math.pow(len(OUTPUT_ALPHABET), size))
    frequencies = [0]*frequencies_size
    for p in range(len(text) - (size-1)):
        pos = 0
        for i in range(size):
            pos += text[p + i] * int(math.pow(len(OUTPUT_ALPHABET), i))
        frequencies[pos] += 1
    for f in np.nonzero(np.array(frequencies))[0]:
        frequencies[f] = frequencies[f] / len(text)
    return before + frequencies


def calculate_ny_gram_frequencies(text, size, interval, recursive=True):
    before = []
    if recursive is True and size > 2:
        before = calculate_ny_gram_frequencies(text, size-1, interval, recursive)
    frequencies_size = int(math.pow(len(OUTPUT_ALPHABET), size))
    frequencies = [0]*frequencies_size
    for p in range(len(text) - (size-1) * interval):
        pos = 0
        for i in range(size):
            pos += text[p + i*interval] * int(math.pow(len(OUTPUT_ALPHABET), i))
        frequencies[pos] += 1
    for f in np.nonzero(np.array(frequencies))[0]:
        frequencies[f] = frequencies[f] / len(text)
    return before + frequencies


def calculate_index_of_coincidence(text):
    n = [0]*len(OUTPUT_ALPHABET)
    for p in text:
        n[p] = n[p] + 1
    coindex = 0
    for i in range(0, len(OUTPUT_ALPHABET)):
        coindex = coindex + n[i] * (n[i] - 1) / len(text) / (len(text) - 1)
    return coindex


def calculate_digraphic_index_of_coincidence(text):
    pair_number = len(OUTPUT_ALPHABET) * len(OUTPUT_ALPHABET)
    n = [0]*pair_number
    for i in range(1, len(text), 1):
        p0, p1 = text[i-1], text[i]
        n[p0 * len(OUTPUT_ALPHABET) + p1] += 1
    coindex = 0
    for i in np.nonzero(np.array(n))[0]:
        coindex += n[i] * (n[i] - 1) / (len(text) - 1) / (len(text) - 2)
    return coindex


def has_letter_j(text):
    return int(OUTPUT_ALPHABET.index(b'j') in text)


def has_hash(text):
    return int(OUTPUT_ALPHABET.index(b'#') in text)


def has_space(text):
    return int(OUTPUT_ALPHABET.index(b' ') in text)


def has_letter_x(text):
    return int(OUTPUT_ALPHABET.index(b'x') in text)


def has_digit_0(text):
    return int(OUTPUT_ALPHABET.index(b'0') in text)


def calculate_chi_square(frequencies):
    english_frequencies = global_english_frequencies
    chi_square = 0
    for i in range(len(frequencies)):
        chi_square = chi_square + (
                    (english_frequencies[i] - frequencies[i]) * (english_frequencies[i] - frequencies[i])) / english_frequencies[i]
    return chi_square / 100


def pattern_repetitions(text):
    counter = 0
    rep = 0
    length = 1
    max_range = False
    for i in range(1, len(text), 1):
        if max_range:
            max_range = False
            continue
        if text[i-1] == text[i]:
            if length == 1:
                counter += 1
            length += 1
        elif length > 1:
            rep += length
            length = 1
        if length == 5:
            rep += length
            length = 1
            max_range = True
    if length > 1:
        rep += length
    if counter != 0:
        return rep / counter
    return 0


def calculate_entropy(text):
    """calculates shannon's entropy index.
    :param text: input numbers-ciphertext
    :return: calculated entropy"""
    # https://stackoverflow.com/questions/2979174/how-do-i-compute-the-approximate-entropy-of-a-bit-string
    _unique, counts = np.unique(text, return_counts=True)
    prob = []
    for c in counts:
        prob.append(float(c) / len(text))
    # calculate the entropy
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
    return entropy / 10


def calculate_autocorrelation(text):
    """calculates the normalized autocorrelation. Currently only working maximal plaintext_size 100. Else 1000 must be raised to length of
    plaintext multiplied by 10.
    :param text: input numbers-ciphertext
    :return: autocorrelation average"""
    # https://stackoverflow.com/questions/14297012/estimate-autocorrelation-using-python
    n = len(text)
    variance = text.var()
    text = text - text.mean()
    r = np.correlate(text, text, mode='full')[-n:]
    result = list(r / (variance * (np.arange(n, 0, -1))))
    result = [float(d) for d in result]
    return result + [0]*(1000-len(text))


def calculate_maximum_index_of_coincidence(text):
    """calculates the maximum IoC for periods 1-15. It is calculated by taking every p-th letter of the ciphertext.
    :see: https://pages.mtu.edu/~shene/NSF-4/Tutorial/VIG/Vig-IOC-Len.html
    :param text: input numbers-ciphertext
    :return: mic"""
    iocs = []
    for i in range(1, 16, 1):
        avg = 0
        for j in range(i):
            txt = []
            for k in range(j, len(text), i):
                txt.append(text[k])
            avg += calculate_index_of_coincidence(txt)
        iocs.append(avg / i)
    return max(iocs)


def calculate_max_kappa(text):
    """calculates the maximum kappa for periods 1-15. It is calculated shifting the ciphertext by p and finding calculating the percentage
    of coinciding characters.
    :param text: input numbers-ciphertext
    :return: mka"""
    mka = []
    for i in range(1, 16, 1):
        shifted_text = []
        for j in range(len(text)-i):
            shifted_text.append(text[(j+i) % len(text)])
        shifted_text += [-1]*i
        mka.append(np.count_nonzero(np.array(shifted_text) == text) / (len(text)-i))
    return max(mka)


def calculate_digraphic_index_of_coincidence_even(text):
    """calculates the digraphic index of coincidence for all characters for with an even index.
    :param text: input numbers-ciphertext
    :return: edi"""
    pair_number = len(OUTPUT_ALPHABET) * len(OUTPUT_ALPHABET)
    n = [0] * pair_number
    for i in range(1, len(text), 2):
        p0, p1 = text[i - 1], text[i]
        n[p0 * len(OUTPUT_ALPHABET) + p1] += 1
    coindex = 0
    for i in np.nonzero(np.array(n))[0]:
        coindex += n[i] * (n[i] - 1) / (len(text) / 2) / (len(text) / 2 - 1)
    return coindex


def calculate_log_digraph_score(text):
    """calculates the log digraph score by averaging all digraph frequencies in the text.
    :param text: input numbers-ciphertext
    :return: ldi"""
    if np.count_nonzero(np.array(text) > 25) > 0:
        return 0
    logdi = global_logdi
    score = 0
    for i in range(len(text)-1):
        if text[i] > 25 or text[i+1] > 26:
            continue
        score += logdi[text[i]][text[i+1]]
    score = score / 10 / (len(text) - 1)
    return score


def calculate_reverse_log_digraph_score(text):
    """calculates the reverse log digraph score by averaging all digraph frequencies in the text.
    :param text: input numbers-ciphertext
    :return: rdi"""
    if np.count_nonzero(np.array(text) > 25) > 0:
        return 0
    logdi = global_logdi
    score = 0
    for i in range(len(text) - 1):
        if text[i] > 25 or text[i + 1] > 26:
            continue
        score += logdi[text[i+1]][text[i]]
    score = score / 10 / (len(text) - 1)
    return score


def calculate_rod_lr(text):
    """calculates the percentage of odd-spaced repeats to all repeats and long repeats.
    :param text: input numbers-ciphertext
    :return: rod, lr"""
    lr = 0
    sum_all = 0
    sum_odd = 0
    for i, c in enumerate(text):
        rep = 0
        for j in range(i+1, len(text), 1):
            if c == text[j]:
                rep += 1
                sum_all += 1
                if (j - i) % 2 == 1:
                    sum_odd += 1
        if rep == 3:
            lr += 1
    lr = math.sqrt(lr) / len(text)
    rod = sum_odd / sum_all
    return rod, lr


def calculate_normal_order(frequencies):
    """calculates the normal order of the text.
    :param frequencies: unigram frequencies of the ciphertext
    :return: nomor"""
    # negate english frequencies to get indices of the highes elements first.
    english_frequencies = global_english_frequencies
    expected = np.array(english_frequencies)
    expected = -expected
    expected = expected.argsort()
    tested = np.array(frequencies)
    tested = -tested
    tested = tested.argsort()
    result = 0
    for i in range(len(expected)):
        result += math.fabs(expected[i] - tested[i])
    return result / 1000


def is_dbl(text):
    """binary value: 1 if text length is even and a doubled character is at an even position, else 0.
    :param text: input numbers-ciphertext
    :return: dbl"""
    if len(text) % 2 != 1:
        for i in range(0, len(text)-1, 2):
            if text[i] == text[i+1]:
                return 1
    return 0


def calculate_nic(text):
    """calculates the maximum Nicodemus IC for periods 3-15.
    :param text: input numbers-ciphertext
    :return: nic"""
    nics = []
    col_len = 5
    for i in range(1, 16, 1):
        ct = [[0] * len(OUTPUT_ALPHABET) for i in range(16)]
        block_len = len(text) // (col_len*i)
        limit = block_len * col_len * i
        index = 0
        for j in range(limit):
            ct[index][text[j]] += 1
            if (j + 1) % col_len == 0:
                index = (index + 1) % i
        z = 0
        for j in range(i):
            x = 0
            y = 0
            for k in range(len(OUTPUT_ALPHABET)):
                x += ct[j][k] * (ct[j][k] - 1)
                y += ct[j][k]
            if y > 1:
                z += x / (y * (y - 1))
        z = z / i
        nics.append(z)
    return max(nics[2:])


def calculate_sdd(text):
    """calculates the average English single letter - digraph discrepancy score.
    :param text: input numbers-ciphertext
    :return: sdd"""
    sdd = global_sdd
    text = list(text)
    score = 0
    for i in range(len(text)-1):
        if text[i] > 25 or text[i+1] > 25:
            continue
        score += sdd[text[i]][text[i+1]]
    score = score / (len(text)-1) / 10
    return score


def calculate_ldi_stats(text):
    """calculates the LDI for Autokey, Beaufort, Porta, Slidefair and Vigenere.
    :param text: input numbers-ciphertext
    :return: a_ldi, b_ldi, p_ldi, s_ldi, v_ldi"""
    if np.count_nonzero(np.array(text) > 25) > 0:
        return [0, 0, 0, 0, 0]
    return ctx.call("get_vig_values", text)


def calculate_ptx(text):
    if np.count_nonzero(np.array(text) > 25) > 0:
        return 0
    return ctx.call("calc_portax_logdi", text)


def calculate_phic(text):
    """calculates the Phillips IC.
    :param text: input numbers-ciphertext
    :return: phic"""
    if np.count_nonzero(np.array(text) > 25) > 0:
        return 0
    combine_alpha = [0,1,2,3,0,4,5,1]
    period = 8
    col_len = 5
    ct = [[0]*26 for _ in range(period-1)]
    block_len = len(text) // (col_len*period)
    limit = block_len*col_len*period
    index = 0
    for i in range(limit):
        ct[combine_alpha[index]][text[i]] += 1
        if (i+1) % col_len == 0:
            index = (index + 1) % period
    z = 0
    for i in range(period - 2):
        x = 0
        y = 0
        for j in range(26):
            x += ct[i][j]*(ct[i][j]-1)
            y += ct[i][j]
        if y > 1:
            z += x / (y * (y - 1))
    z /= (period-2)
    return z*10


def calculate_bdi(text):
    """calculates Max Bifid DIC for periods 3-15.
    :param text: input numbers-ciphertext
    :return: bdi"""
    if np.count_nonzero(np.array(text) > 25) > 0:
        return 0
    best_score = 0
    normalizer = 25*25
    text_len = len(text)
    for period in range(3, 16, 1):
        numb = 0
        freq = [0] * 676
        for i in range(0, text_len, period):
            if i + period < text_len:
                limit = i + period
                second_row = period // 2
            else:
                limit = text_len
                second_row = (text_len-i) // 2
            for j in range(i, limit-second_row, 1):
                freq[text[j]+26*text[j+second_row]] += 1
            numb += limit - second_row - i
        sum_ = 0
        for i in np.nonzero(np.array(freq))[0]:
            sum_ += freq[i] * (freq[i] - 1)
        score = 100*normalizer * sum_ // (numb*(numb-1)) / 1000
        best_score = max(best_score, score)
    return best_score


def calculate_cdd(text):
    """calculates Max Columnar SDD Score for periods 4-15.
    :param text: input numbers-ciphertext
    :return: cdd"""
    if np.count_nonzero(np.array(text) > 25) > 0:
        return 0
    return ctx.call("get_cdd", text)


def calculate_sstd(text):
    """calculates Max STD Score for Swagman periods 4-8.
    :param text: input numbers-ciphertext
    :return: cdd"""
    if np.count_nonzero(np.array(text) > 25) > 0:
        return 0
    return ctx.call("get_sstd", text)


def encrypt(plaintext, label, key_length, keep_unknown_symbols, return_key=False):
    cipher = config.CIPHER_IMPLEMENTATIONS[label]
    plaintext = cipher.filter(plaintext, keep_unknown_symbols)
    key = cipher.generate_random_key(key_length)
    if return_key:
        orig_key = copy.deepcopy(key)
    plaintext_numberspace = map_text_into_numberspace(plaintext, cipher.alphabet, cipher.unknown_symbol_number)
    if isinstance(key, bytes):
        key = map_text_into_numberspace(key, cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 2 and isinstance(key[0], bytes) and isinstance(key[1], bytes):
        key[0] = map_text_into_numberspace(key[0], cipher.alphabet, cipher.unknown_symbol_number)
        key[1] = map_text_into_numberspace(key[1], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 3 and isinstance(key[0], bytes) and isinstance(key[1], bytes) and isinstance(key[2], bytes):
        key[0] = map_text_into_numberspace(key[0], cipher.alphabet, cipher.unknown_symbol_number)
        key[1] = map_text_into_numberspace(key[1], cipher.alphabet, cipher.unknown_symbol_number)
        key[2] = map_text_into_numberspace(key[2], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 2 and isinstance(key[0], bytes) and isinstance(key[1], int):
        key[0] = map_text_into_numberspace(key[0], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 3 and isinstance(key[0], int) and isinstance(key[1], bytes) and isinstance(key[2], bytes):
        key[1] = map_text_into_numberspace(key[1], cipher.alphabet, cipher.unknown_symbol_number)
        key[2] = map_text_into_numberspace(key[2], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 2 and isinstance(key[0], (list, np.ndarray)) and (len(key[0]) == 5 or len(
            key[0]) == 10) and isinstance(key[1], bytes):
        key[1] = map_text_into_numberspace(key[1], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 3 and isinstance(key[0], list) and isinstance(key[1], np.ndarray) and isinstance(
            key[2], bytes):
        key[2] = map_text_into_numberspace(key[2], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, dict):
        new_key_dict = {}
        for k in key:
            new_key_dict[cipher.alphabet.index(k)] = key[k]
        key = new_key_dict

    ciphertext = cipher.encrypt(plaintext_numberspace, key)
    if b'j' not in cipher.alphabet and config.CIPHER_TYPES[label] != 'homophonic':
        ciphertext = normalize_text(ciphertext, 9)
    if b'x' not in cipher.alphabet:
        ciphertext = normalize_text(ciphertext, 23)
    if return_key:
        return ciphertext, orig_key
    return ciphertext


def normalize_text(text, pos):
    for i in range(len(text)):
        if 26 >= text[i] >= pos:
            text[i] += 1
    return text


def calculate_statistics(datum):
    numbers = [int(d) for d in datum]
    unigram_ioc = calculate_index_of_coincidence(numbers)
    digraphic_ioc = calculate_digraphic_index_of_coincidence(numbers)
    # autocorrelation = calculate_autocorrelation(datum)
    frequencies = calculate_frequencies(numbers, 2, recursive=True)

    has_j = has_letter_j(numbers)
    # chi_square = calculate_chi_square(frequencies[0:26])
    # rep = pattern_repetitions(numbers)
    # entropy = calculate_entropy(numbers)
    has_h = has_hash(numbers)
    has_sp = has_space(numbers)
    has_x = has_letter_x(numbers)
    has_0 = has_digit_0(numbers)
    mic = calculate_maximum_index_of_coincidence(numbers)
    mka = calculate_max_kappa(numbers)
    # edi = calculate_digraphic_index_of_coincidence_even(numbers)
    ldi = calculate_log_digraph_score(numbers)
    # rdi = calculate_reverse_log_digraph_score(numbers)
    rod, lr = calculate_rod_lr(numbers)
    nomor = calculate_normal_order(frequencies[0:26])
    # dbl = is_dbl(numbers)
    nic = calculate_nic(numbers)
    sdd = calculate_sdd(numbers)
    ptx = calculate_ptx(numbers)
    phic = calculate_phic(datum)
    bdi = calculate_bdi(numbers)
    # cdd = calculate_cdd(numbers)
    # sstd = calculate_sstd(numbers)
    ldi_stats = calculate_ldi_stats(numbers)

    # baseline model
    # return [unigram_ioc] + [digraphic_ioc] + [has_j] + [entropy] + [chi_square] + [has_h] + [has_sp] + [has_x] + frequencies

    return [unigram_ioc] + [digraphic_ioc] + frequencies + [has_0] + [has_h] + [has_j] + [has_x] + [has_sp] + [rod] + [lr] + [sdd] +\
           [ldi] + [nomor] + [phic] + [bdi] + [ptx] + [nic] + [mka] + [mic] + ldi_stats

    # all features
    # return [unigram_ioc] + [digraphic_ioc] + [has_j] + [entropy] + [chi_square] + [has_h] + [has_sp] + [has_x] + [has_0] + [mic] +\
    #        [mka] + [rep] + [edi] + [ldi] + [rdi] + [rod] + [lr] + [nomor] + [dbl] + [nic] + [sdd] + ldi_stats + [ptx] +\
    #        [phic] + [bdi] + [cdd] + [sstd] + autocorrelation + frequencies

    # all features with maximal calculation time of 3 ms.
    # return [unigram_ioc] + [digraphic_ioc] + [has_j] + [entropy] + [chi_square] + [has_h] + [has_sp] + [has_x] + [has_0] +\
    #        [rep] + [edi] + [ldi] + [rdi] + [rod] + [lr] + [nomor] + [dbl] + [nic] + [sdd] + [ptx] + [phic] + [bdi] +\
    #        [sstd] + autocorrelation + frequencies


def pad_sequences(sequences, maxlen):
    """Pad sequences with data from itself."""
    ret_sequences = []
    for sequence in sequences:
        length = len(sequence)
        sequence = sequence * (maxlen // length) + sequence[:maxlen % length]
        ret_sequences.append(sequence)
    return np.array(ret_sequences)


class TextLine2CipherStatisticsDataset:
    def __init__(self, paths, cipher_types, batch_size, min_text_len, max_text_len, keep_unknown_symbols=False, dataset_workers=None,
                 generate_test_data=False):
        self.keep_unknown_symbols = keep_unknown_symbols
        self.dataset_workers = dataset_workers
        self.cipher_types = cipher_types
        self.batch_size = batch_size
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.epoch = 0
        self.iteration = 0
        self.iter = None
        datasets = []
        for path in paths:
            datasets.append(tf.data.TextLineDataset(path, num_parallel_reads=dataset_workers))
        self.dataset = datasets[0]
        for dataset in datasets[1:]:
            self.dataset = self.dataset.zip(dataset)
        count = 0
        for cipher_t in self.cipher_types:
            index = self.cipher_types.index(cipher_t)
            if isinstance(config.KEY_LENGTHS[index], list):
                count += len(config.KEY_LENGTHS[index])
            else:
                count += 1
        self.key_lengths_count = count
        self.generate_test_data = generate_test_data

    def shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=None):
        new_dataset = copy.copy(self)
        new_dataset.dataset = new_dataset.dataset.shuffle(buffer_size, seed, reshuffle_each_iteration)
        return new_dataset

    def __iter__(self):
        self.iter = self.dataset.__iter__()
        return self

    def __next__(self):
        processes = []
        manager = multiprocessing.Manager()
        c = SimpleSubstitution(config.INPUT_ALPHABET, config.UNKNOWN_SYMBOL, config.UNKNOWN_SYMBOL_NUMBER)
        # debugging does not work here!
        result_list = manager.list()
        if self.generate_test_data:
            ciphertext_list = manager.list()
        for _ in range(self.dataset_workers):
            d = []
            for _ in range(self.batch_size // self.key_lengths_count):
                try:
                    # use the basic prefilter to get the most accurate text length
                    data = c.filter(self.iter.__next__().numpy(), self.keep_unknown_symbols)
                    while len(data) < self.min_text_len:
                        # add the new data to the existing to speed up the searching process.
                        data += c.filter(self.iter.__next__().numpy(), self.keep_unknown_symbols)
                    if len(data) > self.max_text_len != -1:
                        d.append(data[:self.max_text_len-(self.max_text_len % 2)])
                    else:
                        d.append(data[:len(data)-(len(data) % 2)])
                except:
                    self.epoch += 1
                    self.__iter__()
                    data = c.filter(self.iter.__next__().numpy(), self.keep_unknown_symbols)
                    while len(data) < self.min_text_len:
                        data += c.filter(self.iter.__next__().numpy(), self.keep_unknown_symbols)
                    if len(data) > self.max_text_len:
                        d.append(data[:self.max_text_len-(self.max_text_len % 2)])
                    else:
                        d.append(data[:len(data)-(len(data) % 2)])
            if self.generate_test_data:
                process = multiprocessing.Process(target=self._worker, args=(d, result_list, ciphertext_list))
            else:
                process = multiprocessing.Process(target=self._worker, args=(d, result_list))
            process.start()
            processes.append(process)
        if self.generate_test_data:
            return processes, result_list, ciphertext_list
        return processes, result_list

    def _worker(self, data, result, ciphertext_list=None):
        batch = []
        labels = []
        ciphertexts = []
        for d in data:
            for cipher_t in self.cipher_types:
                index = config.CIPHER_TYPES.index(cipher_t)
                label = self.cipher_types.index(cipher_t)
                if isinstance(config.KEY_LENGTHS[label], list):
                    key_lengths = config.KEY_LENGTHS[label]
                else:
                    key_lengths = [config.KEY_LENGTHS[label]]
                for key_length in key_lengths:
                    ciphertext = encrypt(d, index, key_length, self.keep_unknown_symbols)
                    if config.FEATURE_ENGINEERING:
                        statistics = calculate_statistics(ciphertext)
                        batch.append(statistics)
                    else:
                        batch.append(list(ciphertext))
                    if self.generate_test_data:
                        ciphertexts.append(list(ciphertext))
                    labels.append(label)
        if config.PAD_INPUT:
            batch = pad_sequences(batch, maxlen=self.max_text_len)
            batch = batch.reshape(batch.shape[0], batch.shape[1], 1)
        if self.generate_test_data:
            ciphertexts = pad_sequences(ciphertexts, maxlen=self.max_text_len)
            ciphertext_list.append(ciphertexts)
            result.append((batch, labels))
        else:
            result.append((tf.convert_to_tensor(batch), tf.convert_to_tensor(labels)))
