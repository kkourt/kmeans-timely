// vim: set expandtab softtabstop=4 tabstop=4 shiftwidth=4:

// Simple k-means implementation using timely-dataflow
// Kornilios Kourtis <kornilios@gmail.com>

extern crate rand;
extern crate timely;

use rand::distributions::{Normal,IndependentSample};
use std::collections::{HashMap};
use std::f64;

use timely::dataflow::Scope;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::*;
use timely::dataflow::operators::aggregation::*;

pub fn gen_2d_random_normal(n: u32, m: (f64, f64), d: (f64, f64)) -> Vec<(f64,f64)> {
    let norm0 = Normal::new(m.0, d.0);
    let norm1 = Normal::new(m.1, d.1);
    let get_rnd0 = || { norm0.ind_sample(&mut rand::thread_rng())};
    let get_rnd1 = || { norm1.ind_sample(&mut rand::thread_rng())};
    let mut ret = Vec::new();
    for _ in 0..n {
        ret.push((get_rnd0(), get_rnd1()))
    };
    ret
}

pub fn td_kmeans(centroids: Vec<(f64,f64)>) {
    let conf = timely::Configuration::Process(2);
    timely::execute(conf, move |g| {
		let mut cent_i = g.scoped(move |scope| {

            let index = scope.index();
            #[allow(unused_variables)]
            let peers = scope.peers();

            // input for centroids
			let (cent_i, cent_s) = scope.new_input();

            // state
            let mut points = Vec::new();
            let mut centroids = HashMap::new();

            let max_iter = 20;
            let (handle, cycle) = scope.loop_variable(max_iter,1);

            cent_s
            .concat(&cycle) // concat loop edge (results) to initial centroids
            //.inspect(move |x| println!("CENTROIDS: worker: {:?}/{:?} seen: {:?}", index, peers, x))
            .unary_notify(Pipeline,"",vec![],
                move |cent_in, out, notif| {
                    // receive centroids
                    cent_in.for_each(|cap,data| {
                        let vec = centroids.entry(cap.time()).or_insert(vec![]);
                        for d in data.drain(..) {
                            notif.notify_at(cap.clone());
                            vec.push(d);
                        }
                    });
                    // handle end of input
                    notif.for_each(|cap,_,_| {
                        let t = cap.time();

                        // load data points (only the first time)
                        if cap.inner == 0 {
                            assert_eq!(points.len(), 0);
                            points.append(&mut gen_2d_random_normal(40, (180.0,80.0), (10.0,1.0)));
                            points.append(&mut gen_2d_random_normal(60, (160.0,60.0), ( 7.0,2.0)));
                        }

                        // print results
                        if (cap.inner == max_iter) && (index == 0) {
                            for c in centroids.get(&t).unwrap() {
                                println!("=> {:?}", c);
                            }
                        }

                        // find the closest centroid for each point
                        let centroids_t : Vec<(u64, (f64,f64))> = centroids.remove(&t).unwrap();
                        for p in points.iter() {
                            let mut min_dist = f64::MAX;
                            let mut best_centroid = (std::u64::MAX, (f64::NAN, f64::NAN));
                            for (idx, c) in centroids_t.iter().cloned() {
                                let dist = ((c.0-p.0).powi(2) + (c.1-p.1).powi(2)).sqrt();
                                if dist < min_dist {
                                    min_dist = dist;
                                    best_centroid = (idx,c);
                                }
                            }
                            // use an aggregate-friendly format
                            out.session(&cap).give((best_centroid.0, (best_centroid.1, *p)));
                        }

                    })
            })
            //.inspect(move |x| println!("DATA: worker: {:?}/{:?} seen: {:?}", index, peers, x))
            .aggregate::<_,(f64,f64,u64),_,_,_>(
                |_, (_, (px,py)), t | {
                    *t = (t.0 + px, t.1 + py, t.2 + 1);
                },
                |cid, (sum_x, sum_y, cnt) | {
                    let cnt_ = cnt as f64;
                    (cid, (sum_x / cnt_, sum_y / cnt_)) // produce new centroids
                },
                |key | *key
            )
            .broadcast() // make sure that centroids go to all workers
            .connect_loop(handle);

            cent_i
        });

        // initial centroids
		for (c, i) in centroids.iter().zip(0..) {
			cent_i.send((i,*c));
		}
		cent_i.advance_to(1);

    }).unwrap().join();
}

fn main() {
    // initial centroids
    let centroids = vec![(100.0,50.0),(200.0,100.0)];
    td_kmeans(centroids)
}
