using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class size_adjust : MonoBehaviour
{
    public Transform root;
    public Transform hand;
    private float default_dis,current_dis;
    private float default_scale;
    // Start is called before the first frame update
    void Start()
    {
        default_dis= Vector3.Distance(root.position, hand.position);
        default_scale=this.transform.localScale.x;
    }

    // Update is called once per frame
    void Update()
    {
        current_dis=Vector3.Distance(root.position, hand.position);
        float tmp=current_dis/default_dis;
        this.transform.localScale=new Vector3(default_scale*tmp,default_scale*tmp,default_scale*tmp);
    }
}
 
